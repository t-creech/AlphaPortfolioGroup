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

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Establish WRDS Connection
# -------------------------
db = wrds.Connection()  # Ensure your WRDS credentials/environment are set up

# ------------------------ DATA PIPELINE -----------------------------

# -------------------------
# Dataset: AlphaPortfolioData
# -------------------------
class AlphaPortfolioData(Dataset):
    def __init__(self, start_year=2014, end_year=2020, final_year=2016, lookback=12, T = 12):
        """
        Initializes the dataset.
        
        For each asset, we load historical data from CRSP and Compustat via WRDS.
        
        We then build sequential episodes for RL. Each episode has T time steps
        (we set T = lookback, e.g. 12). At each time step t:
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
        """
        Loads CRSP and Compustat data via WRDS.
        Returns:
          merged: the full merged DataFrame (all dates)
          final_data: DataFrame filtered to years >= final_year
        """
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
            crsp_data['mktcap'] = (crsp_data['prc'].abs() * crsp_data['shrout'] * 1000) / 1e6  # In millions
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
        # Query Compustat quarterly data with release dates (rdq)
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
        # Link Compustat GVKEY to CRSP PERMNO
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
        Let T = lookback (i.e. we rebalance for lookback months sequentially).
        For an episode starting at index i, for each time step t (0 <= t < T):
          - The state (for rebalancing time t) is the data from date index i+t to i+t+lookback-1.
          - The one-month forward return is taken from date index i+t+lookback.
        
        This yields:
          - state_seq: (T, num_assets, lookback, num_features)
          - fwd_seq:   (T, num_assets)
          - mask_seq:  (T, num_assets)
        """
        data = self.final_data
        lookback = self.lookback
        T = self.T  # number of rebalancing steps per episode
        unique_dates = pd.to_datetime(data['date'].unique())
        unique_dates_sorted = np.sort(unique_dates)
        num_features = 6  # Using: 'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq', 'atq', 'niq', 'lctq', 'epspiq'
        
        episodes_states = []
        episodes_fwd = []
        episodes_masks = []
        num_episodes = len(unique_dates_sorted) - (2 * lookback) + 1
        logger.info(f"Creating {num_episodes} sequential episodes (T = {T} time steps each).")
        for start_idx in tqdm(range(num_episodes), desc="Creating sequential episodes"):
            episode_states = []  # will have shape (T, global_max_assets, lookback, num_features)
            episode_fwd = []     # will have shape (T, global_max_assets)
            episode_masks = []   # will have shape (T, global_max_assets)
            for t in range(T):
                state_start = start_idx + t
                state_end = state_start + lookback
                fwd_index = state_end  # forward return index for this time step
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
                        # Extract the features: 'permno', 'ret', 'prc', 'vol', 'mktcap', 'saleq', 'atq', 'niq', 'lctq', 'epspiq'
                        features = hist_data[['permno', 'ret', 'prc', 'vol', 'mktcap', 
                                               'saleq']].values
                        step_states[idx] = features
                        step_fwd[idx] = fwd_data['ret'].values[0]
                        step_mask[idx] = True
                episode_states.append(step_states)
                episode_fwd.append(step_fwd)
                episode_masks.append(step_mask)
            # Convert the lists for this episode into arrays.
            episode_states = np.array(episode_states)   # shape: (T, global_max_assets, lookback, num_features)
            episode_fwd = np.array(episode_fwd)           # shape: (T, global_max_assets)
            episode_masks = np.array(episode_masks)       # shape: (T, global_max_assets)
            episodes_states.append(episode_states)
            episodes_fwd.append(episode_fwd)
            episodes_masks.append(episode_masks)
        # Convert the episodes lists to tensors.
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
        # Returns a tuple:
        #   (state_seq, fwd_seq, mask_seq)
        # where:
        #   state_seq: (T, num_assets, lookback, num_features)
        #   fwd_seq:   (T, num_assets)
        #   mask_seq:  (T, num_assets)
        return self.sequences[idx], self.future_returns[idx], self.masks[idx]

# ------------------------- MODEL ARCHITECTURE -------------------------

# -------------------------
# Model: AlphaPortfolioModel (SREM + CAAN + Portfolio Generator)
# -------------------------

class AlphaPortfolioModel(nn.Module):

    def __init__(self, num_features, lookback, d_model=32, nhead=4, num_encoder_layers=2, d_attn=16, G=5):
        """
        The model processes an asset's historical state (shape: (num_assets, lookback, num_features))
        and produces a portfolio weight vector (one weight per asset).
        It uses:
          - An input projection to embed each time step.
          - A Transformer encoder (SREM) to capture temporal dependencies.
          - A CAAN module to compute inter-asset relationships.
          - A portfolio generator that selects top and bottom G assets.
        """
        super().__init__()
        self.G = G
        self.lookback = lookback
        self.d_model = d_model

        # Log the initialization.
        logger.info(f"Initializing AlphaPortfolioModel with num_features={num_features}, lookback={lookback}, d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, d_attn={d_attn}, G={G}")
        
        # Project raw features into embedding space.
        self.input_projection = nn.Linear(num_features, d_model)
        logger.info(f"Input projection layer: {self.input_projection}")

        # SREM: Transformer Encoder for sequence representation.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        logger.info("Transformer encoder initialized.")

        # After encoding, we flatten the sequence to obtain an asset representation r.
        self.r_dim = lookback * d_model
        logger.info(f"Asset representation dimension (r_dim): {self.r_dim}")

        # CAAN: linear layers to compute Query, Key, Value from asset representation.
        self.query_layer = nn.Linear(self.r_dim, d_attn)
        self.key_layer = nn.Linear(self.r_dim, d_attn)
        self.value_layer = nn.Linear(self.r_dim, d_attn)
        logger.info("CAAN layers (query, key, value) initialized.")

        # Winner score: project aggregated attention vector into a score.
        self.score_layer = nn.Linear(d_attn, 1)
        logger.info("Score layer for winner scores initialized.")

    def forward(self, x, mask):
        """
        Input:
          x: Tensor of shape (B, num_assets, lookback, num_features)
          mask: Tensor of shape (B, num_assets)
        Returns:
          portfolio_weights: Tensor of shape (B, num_assets)
          winner_scores: Tensor of shape (B, num_assets)
        Detailed logging is provided at each step.
        """
        logger.info(f"[Model] Input x shape: {x.shape}")
        B, A, L, feat_dim = x.size()
        # Flatten the assets dimension to process each asset's sequence independently.
        x = x.view(B * A, L, feat_dim)  # (B*A, L, feat_dim)
        logger.info(f"[Model] After flattening: {x.shape}")
        
        # Project raw features into embeddings.
        x = self.input_projection(x)    # (B*A, L, d_model)
        logger.info(f"[Model] After input projection: {x.shape}")
        
        # Transpose for the Transformer: required shape (L, B*A, d_model)
        x = x.transpose(0, 1)           # (L, B*A, d_model)
        logger.info(f"[Model] After transpose for Transformer: {x.shape}")
        
        # Pass through the Transformer encoder.
        encoded = self.transformer_encoder(x)  # (L, B*A, d_model)
        logger.info(f"[Model] After Transformer encoder: {encoded.shape}")
        
        # Transpose back to (B*A, L, d_model)
        encoded = encoded.transpose(0, 1)  # (B*A, L, d_model)
        logger.info(f"[Model] After transpose back: {encoded.shape}")
        
        # Flatten the time dimension: each asset gets a single representation vector.
        asset_repr = encoded.contiguous().view(B, A, -1)  # (B, A, L*d_model)
        logger.info(f"[Model] Asset representation shape: {asset_repr.shape}")
        
        # CAAN: compute query, key, and value vectors.
        Q = self.query_layer(asset_repr)  # (B, A, d_attn)
        K = self.key_layer(asset_repr)      # (B, A, d_attn)
        V = self.value_layer(asset_repr)    # (B, A, d_attn)
        logger.info(f"[Model] Query shape: {Q.shape}, Key shape: {K.shape}, Value shape: {V.shape}")
        
        d_attn = Q.size(-1)
        # Compute inter-asset attention scores.
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_attn)  # (B, A, A)
        logger.info(f"[Model] Attention scores shape: {scores.shape}")
        
        # Mask out invalid assets: for assets with mask=False, set scores to a large negative number.
        mask_float = mask.float()           # (B, A)
        mask_exp = mask_float.unsqueeze(1)    # (B, 1, A)
        scores = scores + (1 - mask_exp) * (-1e9)
        logger.info("[Model] Applied mask to attention scores.")
        
        # Softmax over assets (for each asset i, normalize over j).
        attn_weights = F.softmax(scores, dim=-1)  # (B, A, A)
        logger.info(f"[Model] Attention weights shape: {attn_weights.shape}")
        
        # Aggregate value vectors: compute the attention vector for each asset.
        attn_vec = torch.bmm(attn_weights, V)  # (B, A, d_attn)
        logger.info(f"[Model] Aggregated attention vector shape: {attn_vec.shape}")
        
        # Compute winner scores using a fully connected layer and tanh activation.
        winner_scores = torch.tanh(self.score_layer(attn_vec)).squeeze(-1)  # (B, A)
        logger.info(f"[Model] Winner scores shape (pre-mask): {winner_scores.shape}")
        # For invalid assets, set the winner score to a very low value.
        winner_scores = winner_scores.masked_fill(~mask, -1e9)
        logger.info("[Model] Applied mask to winner scores.")
        
        # -------------------------
        # Portfolio Generation:
        # For each batch element, select top G for long positions and bottom G for short positions.
        # -------------------------
        portfolio_weights = []
        for i in range(B):
            scores_i = winner_scores[i]  # (A,)
            valid_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
            logger.info(f"[Model] Batch {i}: valid asset indices: {valid_idx}")
            if valid_idx.numel() == 0:
                portfolio_weights.append(torch.zeros_like(scores_i))
                continue
            valid_scores = scores_i[valid_idx]
            G = self.G
            # Adjust G if not enough valid assets.
            G_adj = min(G, valid_scores.size(0) // 2) if valid_scores.size(0) >= 2 else 1
            logger.info(f"[Model] Batch {i}: G_adj = {G_adj}")

            # For long positions: top G_adj highest scores.
            sorted_long = torch.argsort(valid_scores, descending=True)
            top_indices = valid_idx[sorted_long[:G_adj]]
            logger.info(f"[Model] Batch {i}: top_indices for long positions: {top_indices}")
            # For short positions: bottom G_adj lowest scores.
            sorted_short = torch.argsort(valid_scores, descending=False)
            bottom_indices = valid_idx[sorted_short[:G_adj]]
            logger.info(f"[Model] Batch {i}: bottom_indices for short positions: {bottom_indices}")
            
            # Compute long weights (softmax over top scores).
            long_scores = scores_i[top_indices]
            long_weights = torch.softmax(long_scores, dim=0)
            logger.info(f"[Model] Batch {i}: long_weights: {long_weights}")
            # Compute short weights (softmax over negative bottom scores).
            short_scores = scores_i[bottom_indices]
            short_weights = torch.softmax(-short_scores, dim=0)
            logger.info(f"[Model] Batch {i}: short_weights: {short_weights}")
            
            b = torch.zeros_like(scores_i)
            b[top_indices] = long_weights
            b[bottom_indices] = -short_weights  # negative for short positions
            logger.info(f"[Model] Batch {i}: portfolio weights: {b}")
            portfolio_weights.append(b)
        portfolio_weights = torch.stack(portfolio_weights, dim=0)  # (B, A)
        logger.info(f"[Model] Final portfolio_weights shape: {portfolio_weights.shape}")
        return portfolio_weights, winner_scores


# ------------------------- TRAINING PIPELINE -------------------------

# -------------------------
# Training Loop (RL-style Sequential Rebalancing)
# -------------------------

def train_model_sequential(dataset, model, num_epochs=10, learning_rate=1e-4, device='cpu', batch_size=1):
    """
    Each episode consists of T sequential rebalancing steps.
    For each step t (0 <= t < T):
      - Get the state (of shape: (num_assets, lookback, num_features))
      - Compute portfolio weights for that month.
      - Get the one-month forward returns (of shape: (num_assets,)) for that month.
      - Compute the portfolio return (dot product) for that month.
    After T steps, compute the Sharpe ratio of the T monthly returns as the delayed reward.
    Detailed logging is included at every step.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        logger.info(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
        epoch_sharpes = []
        for episode_idx, (state_seq, fwd_seq, mask_seq) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # state_seq: (B, T, A, lookback, num_features)
            # fwd_seq: (B, T, A)
            # mask_seq: (B, T, A)
            logger.info(f"[Episode {episode_idx}] state_seq shape: {state_seq.shape}, fwd_seq shape: {fwd_seq.shape}, mask_seq shape: {mask_seq.shape}")
            
            # We'll process each episode sequentially (time steps t=0 to T-1)
            B, T, A, L, F = state_seq.size()
            monthly_returns = []
            for t in range(T):
                state_t = state_seq[:, t, :, :, :]   # shape: (B, A, L, F)
                fwd_t = fwd_seq[:, t, :]              # shape: (B, A)
                mask_t = mask_seq[:, t, :]            # shape: (B, A)
                logger.info(f"[Episode {episode_idx}][Time {t}] state_t shape: {state_t.shape}, fwd_t shape: {fwd_t.shape}")
                
                # Compute portfolio weights for this time step.
                portfolio_weights, winner_scores = model(state_t, mask_t)  # portfolio_weights: (B, A)
                logger.info(f"[Episode {episode_idx}][Time {t}] portfolio_weights: {portfolio_weights}")
                logger.info(f"[Episode {episode_idx}][Time {t}] winner_scores: {winner_scores}")
                
                # Compute portfolio return for time t.
                # Expand portfolio_weights to (B, A, 1) so that we can multiply elementwise with fwd_t (B, A)
                # Since fwd_t is a scalar per asset for that month, do elementwise multiplication and sum over assets.
                period_return = (portfolio_weights * fwd_t).sum(dim=1)  # (B,)
                logger.info(f"[Episode {episode_idx}][Time {t}] period_return: {period_return}")
                monthly_returns.append(period_return.squeeze(0))  # assuming B=1; adjust accordingly
            
            # After T steps, stack monthly returns to get a tensor of shape (T,)
            monthly_returns = torch.stack(monthly_returns)  # (T,)
            logger.info(f"[Episode {episode_idx}] Monthly returns: {monthly_returns}")
            
            # Compute Sharpe ratio for the episode.
            mean_return = monthly_returns.mean()
            std_return = monthly_returns.std()
            sharpe_ratio = mean_return / (std_return + 1e-6)
            logger.info(f"[Episode {episode_idx}] Episode Sharpe Ratio: {sharpe_ratio}")
            
            loss = -sharpe_ratio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_sharpes.append(sharpe_ratio.item())
        
        avg_sharpe = np.mean(epoch_sharpes)
        logger.info(f"Epoch {epoch+1}: Average Sharpe Ratio = {avg_sharpe:.4f}")


# ------------------------- MAIN SCRIPT -------------------------


# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":

    lookback = 12
    start_year = 2015
    final_year = 2017
    end_year = 2020
    T = 12  # number of rebalancing steps (months) per episode
    model_G = 3        # number of assets selected for long and short in portfolio generation
    batch_size = 1     # we process one episode at a time for clarity in logging
    num_epochs = 10     # set to a small number for demonstration

    # Initialize the dataset.
    dataset = AlphaPortfolioData(start_year=start_year, end_year=end_year, final_year=final_year, lookback=lookback, T=T)
    logger.info(f"Dataset contains {dataset.sequences.shape[0]} episodes, each with {dataset.sequences.shape[1]} time steps, {dataset.sequences.shape[2]} assets.")

    num_features = dataset.sequences.shape[-1]

    # Initialize the model.
    model = AlphaPortfolioModel(num_features=num_features, lookback=lookback,
                                d_model=16, nhead=2, num_encoder_layers=1, d_attn=8, G=model_G)

    # Determine device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Train the model using sequential rebalancing.
    train_model_sequential(dataset, model, num_epochs=num_epochs, learning_rate=1e-4, device=device, batch_size=batch_size)
