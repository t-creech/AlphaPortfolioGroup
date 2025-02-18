import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import logging
import wrds
import matplotlib.pyplot as plt
import os

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_file_logger(log_file):
    """Add a file handler to the logger."""
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# Set up a file logger.
log_filename = "training.log"
setup_file_logger(log_filename)
logger.info("File logger initialized.")

# -------------------------
# Establish WRDS Connection
# -------------------------
db = wrds.Connection()  # Ensure your WRDS credentials/environment are set up

# ------------------------ DATA PIPELINE -----------------------------
class AlphaPortfolioData(Dataset):
    def __init__(self, start_year=2014, end_year=2020, final_year=2016, lookback=12, T=12):
        """
        Initializes the dataset.
        
        For each asset, we load historical data from CRSP and Compustat via WRDS.
        
        We then build sequential episodes for RL. Each episode has T time steps.
        At each time step t:
          - The state is the historical window of length 'lookback' for all assets.
            (Shape: [num_assets, lookback, num_features])
          - The forward (one‐month) return for each asset is extracted.
            (Shape: [num_assets])
          - A mask indicates whether the asset has complete data.
        
        Overall, each episode is composed of:
          - state_seq: (T, num_assets, lookback, num_features)
          - fwd_seq: (T, num_assets)
          - mask_seq: (T, num_assets)
        """
        super().__init__()
        self.lookback = lookback
        self.T = T
        self.merged, self.final_data = self._load_wrds_data(start_year, end_year, final_year)
        self.unique_permnos = sorted(self.final_data['permno'].unique())
        self.global_max_assets = len(self.unique_permnos)
        self.permno_to_idx = {permno: idx for idx, permno in enumerate(self.unique_permnos)}
        self.sequences, self.future_returns, self.masks = self._create_sequences()
        logger.info(f"Dataset initialized: {len(self.sequences)} episodes created.")
        if len(self.sequences) > 0:
            logger.info(f"Example episode state shape: {self.sequences[0].shape} "
                        f"(T, num_assets, lookback, num_features)")
            logger.info(f"Example episode future returns shape: {self.future_returns[0].shape} "
                        f"(T, num_assets)")
            logger.info(f"Example episode mask shape: {self.masks[0].shape} "
                        f"(T, num_assets)")

    def _load_wrds_data(self, start_year, end_year, final_year):
        permno_list = []
        combined_data = pd.DataFrame()
        for year in range(start_year, end_year+1):
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            crsp_query = f"""
                SELECT a.permno, a.date, a.ret, a.prc, a.shrout, 
                       a.vol, a.cfacshr, a.altprc, a.retx
                FROM crsp.msf AS a
                WHERE a.date BETWEEN '{start_date}' AND '{end_date}'
                  AND a.permno IN (
                      SELECT permno FROM crsp.msenames 
                      WHERE exchcd BETWEEN 1 AND 3  
                        AND shrcd IN (10, 11)
                  )
                """
            crsp_data = db.raw_sql(crsp_query)
            query_ticker = """
                SELECT permno, namedt, nameenddt, ticker
                FROM crsp.stocknames
            """
            stocknames = db.raw_sql(query_ticker)
            crsp_data = crsp_data.merge(stocknames.drop_duplicates(subset=['permno']), on='permno', how='left')
            crsp_data = crsp_data.dropna(subset=['ticker'])
            crsp_data['mktcap'] = (crsp_data['prc'].abs() * crsp_data['shrout'] * 1000) / 1e6
            crsp_data['year'] = pd.to_datetime(crsp_data['date']).dt.year
            crsp_data = crsp_data.dropna(subset=['mktcap'])
            top_50_permnos_by_year = crsp_data.groupby('permno')['mktcap'].agg(['max']).reset_index()\
                                     .sort_values(by='max', ascending=False).head(50)['permno'].unique()
            permno_list.extend(top_50_permnos_by_year)
            combined_data = pd.concat([combined_data, crsp_data[crsp_data['permno'].isin(permno_list)]], axis=0)
        combined_data = combined_data[['permno', 'ticker', 'date', 'ret', 'prc', 'shrout', 'vol', 'mktcap', 'year']]
        combined_data['date'] = pd.to_datetime(combined_data['date'])

        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-31'
        fund_query = f"""
            SELECT gvkey, datadate, rdq, saleq
            FROM comp.fundq
            WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
              AND datadate BETWEEN '{start_date}' AND '{end_date}'
              AND rdq IS NOT NULL
            """
        fund = db.raw_sql(fund_query)
        fund['rdq'] = pd.to_datetime(fund['rdq'])
        fund['datadate'] = pd.to_datetime(fund['datadate'])
        link_query = """
            SELECT lpermno AS permno, gvkey, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
            """
        link = db.raw_sql(link_query)
        fund = pd.merge(fund, link, on='gvkey', how='left')
        fund = fund.dropna(subset=['permno'])
        combined_data_sorted = combined_data.sort_values('date')
        fund_sorted = fund.sort_values('rdq')
        fund_sorted['permno'] = fund_sorted['permno'].astype(int)
        merged = pd.merge_asof(
            combined_data_sorted,
            fund_sorted,
            left_on='date',
            right_on='rdq',
            by='permno',
            direction='backward'
        )
        merged = merged.sort_values(by='date')
        merged = merged[['permno', 'ticker', 'date', 'ret', 'prc','vol', 'mktcap', 
                         'gvkey', 'rdq', 'saleq']]
        merged = merged.ffill()
        unique_dates = merged['date'].unique()
        date_mapping = {date: i for i, date in enumerate(sorted(unique_dates))}
        merged['date_mapped'] = merged['date'].map(date_mapping)
        merged['year'] = pd.to_datetime(merged['date']).dt.year
        final_data = merged[merged['year'] >= final_year]
        logger.info(f"Data loaded: merged shape {merged.shape}, final_data shape {final_data.shape}")
        return merged, final_data

    def _create_sequences(self):
        """
        Creates sequential episodes for RL.
        
        For each episode, we use a sliding window over the sorted unique dates.
        Let T = self.T (number of rebalancing steps per episode).
        For an episode starting at index i, for each time step t (0 <= t < T):
          - The state is the data from date index i+t to i+t+lookback-1.
          - The one-month forward return is taken from date index i+t+lookback.
        
        This yields:
          - state_seq: (T, num_assets, lookback, num_features)
          - fwd_seq:   (T, num_assets)
          - mask_seq:  (T, num_assets)
        """
        data = self.final_data
        lookback = self.lookback
        T = self.T
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_dates_sorted = np.sort(unique_dates)
        num_features = 6  # Here we use: 'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq'
        
        episodes_states = []
        episodes_fwd = []
        episodes_masks = []
        num_episodes = len(unique_dates_sorted) - (2 * lookback) + 1
        logger.info(f"Creating {num_episodes} sequential episodes (T = {T} time steps each).")
        for start_idx in tqdm(range(num_episodes), desc="Creating sequential episodes"):
            episode_states = []  # shape: (T, global_max_assets, lookback, num_features)
            episode_fwd = []     # shape: (T, global_max_assets)
            episode_masks = []   # shape: (T, global_max_assets)
            for t in range(T):
                state_start = start_idx + t
                state_end = state_start + lookback
                fwd_index = state_end
                step_states = np.zeros((self.global_max_assets, lookback, num_features))
                step_fwd = np.zeros((self.global_max_assets,))
                step_mask = np.zeros((self.global_max_assets,), dtype=bool)
                for permno in self.unique_permnos:
                    idx = self.permno_to_idx[permno]
                    hist_data = data[
                        (data['permno'] == permno) &
                        (data['date'] >= unique_dates_sorted[state_start]) &
                        (data['date'] < unique_dates_sorted[state_end])
                    ].sort_values('date')
                    fwd_data = data[
                        (data['permno'] == permno) &
                        (data['date'] == unique_dates_sorted[fwd_index])
                    ]
                    if len(hist_data) == lookback and len(fwd_data) == 1:
                        features = hist_data[['permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq']].values
                        step_states[idx] = features
                        step_fwd[idx] = fwd_data['ret'].values[0]
                        step_mask[idx] = True
                episode_states.append(step_states)
                episode_fwd.append(step_fwd)
                episode_masks.append(step_mask)
            episode_states = np.array(episode_states)   # (T, global_max_assets, lookback, num_features)
            episode_fwd = np.array(episode_fwd)           # (T, global_max_assets)
            episode_masks = np.array(episode_masks)       # (T, global_max_assets)
            episodes_states.append(episode_states)
            episodes_fwd.append(episode_fwd)
            episodes_masks.append(episode_masks)
        sequences_tensor = torch.tensor(np.array(episodes_states), dtype=torch.float32)
        future_returns_tensor = torch.tensor(np.array(episodes_fwd), dtype=torch.float32)
        masks_tensor = torch.tensor(np.array(episodes_masks), dtype=torch.bool)
        logger.info(f"Created sequences tensor shape: {sequences_tensor.shape}")
        logger.info(f"Created future_returns tensor shape: {future_returns_tensor.shape}")
        logger.info(f"Created masks tensor shape: {masks_tensor.shape}")
        return sequences_tensor, future_returns_tensor, masks_tensor

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.future_returns[idx], self.masks[idx]

# ------------------------- MODEL ARCHITECTURE -------------------------
class AlphaPortfolioModel(nn.Module):
    def __init__(self, num_features, lookback, d_model=32, nhead=4, num_encoder_layers=2, d_attn=16, G=5):
        """
        Processes an asset's historical state (shape: (num_assets, lookback, num_features))
        and produces a portfolio weight vector (one weight per asset).
        Uses:
          - Input projection (Linear) to embed each time step.
          - A Transformer encoder (SREM) for temporal dependencies.
          - A CAAN module for inter-asset relationships.
          - A portfolio generator that selects top and bottom G assets.
        """
        super().__init__()
        self.G = G
        self.lookback = lookback
        self.d_model = d_model

        logger.info(f"Initializing AlphaPortfolioModel with num_features={num_features}, lookback={lookback}, "
                    f"d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, "
                    f"d_attn={d_attn}, G={G}")
        
        self.input_projection = nn.Linear(num_features, d_model)
        logger.info(f"Input projection layer: {self.input_projection}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        logger.info("Transformer encoder initialized.")

        self.r_dim = lookback * d_model
        logger.info(f"Asset representation dimension (r_dim): {self.r_dim}")

        self.query_layer = nn.Linear(self.r_dim, d_attn)
        self.key_layer = nn.Linear(self.r_dim, d_attn)
        self.value_layer = nn.Linear(self.r_dim, d_attn)
        logger.info("CAAN layers (query, key, value) initialized.")

        self.score_layer = nn.Linear(d_attn, 1)
        logger.info("Score layer for winner scores initialized.")

    def forward(self, x, mask):
        logger.info(f"[Model] Input x shape: {x.shape}")
        B, A, L, feat_dim = x.size()
        x = x.view(B * A, L, feat_dim)
        logger.info(f"[Model] After flattening: {x.shape}")
        
        x = self.input_projection(x)
        logger.info(f"[Model] After input projection: {x.shape}")
        
        x = x.transpose(0, 1)
        logger.info(f"[Model] After transpose for Transformer: {x.shape}")
        
        encoded = self.transformer_encoder(x)
        logger.info(f"[Model] After Transformer encoder: {encoded.shape}")
        
        encoded = encoded.transpose(0, 1)
        logger.info(f"[Model] After transpose back: {encoded.shape}")
        
        asset_repr = encoded.contiguous().view(B, A, -1)
        logger.info(f"[Model] Asset representation shape: {asset_repr.shape}")
        
        Q = self.query_layer(asset_repr)
        K = self.key_layer(asset_repr)
        V = self.value_layer(asset_repr)
        logger.info(f"[Model] Query shape: {Q.shape}, Key shape: {K.shape}, Value shape: {V.shape}")
        
        d_attn = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_attn)
        logger.info(f"[Model] Attention scores shape: {scores.shape}")
        
        mask_float = mask.float()
        mask_exp = mask_float.unsqueeze(1)
        scores = scores + (1 - mask_exp) * (-1e9)
        logger.info("[Model] Applied mask to attention scores.")
        
        attn_weights = F.softmax(scores, dim=-1)
        logger.info(f"[Model] Attention weights shape: {attn_weights.shape}")
        
        attn_vec = torch.bmm(attn_weights, V)
        logger.info(f"[Model] Aggregated attention vector shape: {attn_vec.shape}")
        
        winner_scores = torch.tanh(self.score_layer(attn_vec)).squeeze(-1)
        logger.info(f"[Model] Winner scores shape (pre-mask): {winner_scores.shape}")
        winner_scores = winner_scores.masked_fill(~mask, -1e9)
        logger.info("[Model] Applied mask to winner scores.")
        
        portfolio_weights = []
        for i in range(B):
            scores_i = winner_scores[i]
            valid_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
            logger.info(f"[Model] Batch {i}: valid asset indices: {valid_idx}")
            if valid_idx.numel() == 0:
                portfolio_weights.append(torch.zeros_like(scores_i))
                continue
            valid_scores = scores_i[valid_idx]
            G = self.G
            G_adj = min(G, valid_scores.size(0) // 2) if valid_scores.size(0) >= 2 else 1
            logger.info(f"[Model] Batch {i}: G_adj = {G_adj}")

            sorted_long = torch.argsort(valid_scores, descending=True)
            top_indices = valid_idx[sorted_long[:G_adj]]
            logger.info(f"[Model] Batch {i}: top_indices for long positions: {top_indices}")
            sorted_short = torch.argsort(valid_scores, descending=False)
            bottom_indices = valid_idx[sorted_short[:G_adj]]
            logger.info(f"[Model] Batch {i}: bottom_indices for short positions: {bottom_indices}")
            
            long_scores = scores_i[top_indices]
            long_weights = torch.softmax(long_scores, dim=0)
            logger.info(f"[Model] Batch {i}: long_weights: {long_weights}")
            short_scores = scores_i[bottom_indices]
            short_weights = torch.softmax(-short_scores, dim=0)
            logger.info(f"[Model] Batch {i}: short_weights: {short_weights}")
            
            b = torch.zeros_like(scores_i)
            b[top_indices] = long_weights
            b[bottom_indices] = -short_weights
            logger.info(f"[Model] Batch {i}: portfolio weights: {b}")
            portfolio_weights.append(b)
        portfolio_weights = torch.stack(portfolio_weights, dim=0)
        logger.info(f"[Model] Final portfolio_weights shape: {portfolio_weights.shape}")
        return portfolio_weights, winner_scores

# ------------------------- PLOTTING FUNCTIONS -------------------------
def plot_epoch_sharpe(avg_sharpes, save_path):
    plt.figure()
    epochs = list(range(1, len(avg_sharpes)+1))
    plt.plot(epochs, avg_sharpes, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Sharpe Ratio')
    plt.title('Convergence of Average Sharpe Ratio over Epochs')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved epoch convergence plot at {save_path}")

def plot_episode_sharpe(episode_sharpes, epoch, save_path):
    plt.figure()
    episodes = list(range(1, len(episode_sharpes)+1))
    plt.plot(episodes, episode_sharpes, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Episode Sharpe Ratio')
    plt.title(f'Episode-wise Sharpe Ratios for Epoch {epoch}')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved episode Sharpe plot for epoch {epoch} at {save_path}")

# ------------------------- TRAINING PIPELINE -------------------------
def train_model_sequential(dataset, model, num_epochs=10, learning_rate=1e-4, device='cpu', batch_size=1, plots_dir='plots'):
    """
    Each episode consists of T sequential rebalancing steps.
    For each step t:
      - Get the state (shape: (num_assets, lookback, num_features))
      - Compute portfolio weights for that time step.
      - Get the one-month forward returns (shape: (num_assets,)) for that time step.
      - Compute the portfolio return (dot product) for that time step.
    After T steps, compute the Sharpe ratio of the T monthly returns as the delayed reward.
    """
    os.makedirs(plots_dir, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    all_epoch_avg_sharpes = []
    
    for epoch in range(num_epochs):
        logger.info(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
        epoch_sharpes = []
        for episode_idx, (state_seq, fwd_seq, mask_seq) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # state_seq: (B, T, A, lookback, num_features)
            # fwd_seq: (B, T, A)
            # mask_seq: (B, T, A)
            logger.info(f"[Episode {episode_idx}] state_seq shape: {state_seq.shape}, "
                        f"fwd_seq shape: {fwd_seq.shape}, mask_seq shape: {mask_seq.shape}")
            
            B, T, A, L, F = state_seq.size()
            episode_episode_sharpes = []  # To store each episode's Sharpe ratio
            monthly_returns = []
            for t in range(T):
                state_t = state_seq[:, t, :, :, :]   # (B, A, L, F)
                fwd_t = fwd_seq[:, t, :]              # (B, A)
                mask_t = mask_seq[:, t, :]            # (B, A)
                logger.info(f"[Episode {episode_idx}][Time {t}] state_t shape: {state_t.shape}, fwd_t shape: {fwd_t.shape}")
                
                portfolio_weights, winner_scores = model(state_t, mask_t)
                logger.info(f"[Episode {episode_idx}][Time {t}] portfolio_weights: {portfolio_weights}")
                logger.info(f"[Episode {episode_idx}][Time {t}] winner_scores: {winner_scores}")
                
                period_return = (portfolio_weights * fwd_t).sum(dim=1)  # (B,)
                logger.info(f"[Episode {episode_idx}][Time {t}] period_return: {period_return}")
                monthly_returns.append(period_return.squeeze(0))  # assuming B=1
            
            monthly_returns = torch.stack(monthly_returns)  # (T,)
            logger.info(f"[Episode {episode_idx}] Monthly returns: {monthly_returns}")
            
            mean_return = monthly_returns.mean()
            std_return = monthly_returns.std()
            sharpe_ratio = mean_return / (std_return + 1e-6)
            logger.info(f"[Episode {episode_idx}] Episode Sharpe Ratio: {sharpe_ratio}")
            
            loss = -sharpe_ratio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_sharpes.append(sharpe_ratio.item())
            episode_episode_sharpes.append(sharpe_ratio.item())
        
        epoch_avg = np.mean(epoch_sharpes)
        all_epoch_avg_sharpes.append(epoch_avg)
        logger.info(f"Epoch {epoch+1}: Average Sharpe Ratio = {epoch_avg:.4f}")
    
    # After all epochs, plot the convergence of epoch-average Sharpe ratios.
    plot_epoch_path = os.path.join(plots_dir, "epoch_convergence.png")
    plot_epoch_sharpe(all_epoch_avg_sharpes, plot_epoch_path)
    logger.info("Training complete.")

# ------------------------- MAIN SCRIPT -------------------------
if __name__ == "__main__":
    lookback = 12
    start_year = 2015
    final_year = 2017
    end_year = 2020
    T = 12  # number of rebalancing steps per episode
    model_G = 3  # number of assets selected for long/short in portfolio generation
    batch_size = 1
    num_epochs = 30

    dataset = AlphaPortfolioData(start_year=start_year, end_year=end_year, final_year=final_year, lookback=lookback, T=T)
    logger.info(f"Dataset contains {dataset.sequences.shape[0]} episodes, each with {dataset.sequences.shape[1]} time steps, {dataset.sequences.shape[2]} assets.")

    num_features = dataset.sequences.shape[-1]

    model = AlphaPortfolioModel(num_features=num_features, lookback=lookback,
                                d_model=16, nhead=2, num_encoder_layers=1, d_attn=8, G=model_G)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_model_sequential(dataset, model, num_epochs=num_epochs, learning_rate=1e-4, device=device, batch_size=batch_size, plots_dir='plots')
