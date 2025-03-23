from imports import *
from model_architecture import *
from plotting_functions_for_convergence import *
from data_pipeline import *
from training_model import *
import warnings
warnings.filterwarnings("ignore")


class AlphaPortfolioEvaluator:
    def __init__(self, dataset, model, device='cpu', G = 5):
        """
        Initializes the evaluator with a dataset and a trained model.
        
        Args:
            dataset: A PyTorch Dataset instance (e.g., AlphaPortfolioData) where each sample is a tuple
                     (state_seq, fwd_seq, mask_seq). fwd_seq and mask_seq have shape (T, num_assets).
            model: The trained portfolio model.
            device: 'cpu' or 'cuda'.
            G: Number of assets to long and short in the baseline.
        """
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.G = G

    def test_model(self):
        """
        Tests the trained model on the dataset.
        
        For each episode in the dataset:
         - Iterates over T time steps.
         - At each step, runs a forward pass (using the given state and mask) to obtain portfolio weights.
         - Computes the period return as the dot product of portfolio weights and the forward returns.
         - Computes the episode Sharpe ratio from the monthly returns.
         
        Returns:
            all_episode_sharpes: List of Sharpe ratios (one per episode).
        """
        self.model.eval()  # set model to evaluation mode
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        all_episode_sharpes = []
        
        with torch.no_grad():
            for episode_idx, (state_seq, fwd_seq, mask_seq) in enumerate(dataloader):
                # state_seq: (B, T, A, lookback, num_features)
                # fwd_seq:   (B, T, A)
                # mask_seq:  (B, T, A)
                state_seq = state_seq.to(self.device)
                fwd_seq = fwd_seq.to(self.device)
                mask_seq = mask_seq.to(self.device)
                
                B, T, A, L, F = state_seq.size()
                monthly_returns = []
                for t in range(T):
                    # Extract data for time step t.
                    state_t = state_seq[:, t, :, :, :]
                    fwd_t   = fwd_seq[:, t, :]
                    mask_t  = mask_seq[:, t, :]
                    # Forward pass through the model.
                    portfolio_weights, _ = self.model(state_t, mask_t)
                    # Compute period return (assuming B = 1).
                    period_return = (portfolio_weights * fwd_t).sum(dim=1)
                    monthly_returns.append(period_return.squeeze(0))
                
                monthly_returns = torch.stack(monthly_returns)  # shape (T,)
                mean_return = monthly_returns.mean()
                std_return = monthly_returns.std()
                sharpe_ratio = mean_return / (std_return + 1e-6)
                all_episode_sharpes.append(sharpe_ratio.item())
                logger.info(f"Episode {episode_idx}: Sharpe Ratio = {sharpe_ratio.item():.4f}")
        
        avg_sharpe = np.mean(all_episode_sharpes)
        logger.info(f"Average Sharpe Ratio over test episodes: {avg_sharpe:.4f}")
        return avg_sharpe

    @staticmethod
    def compute_sharpe_ratio(returns):
        """
        Computes the Sharpe ratio from a 1D array of returns.
        
        Args:
            returns: 1D numpy array or list of periodic returns.
        
        Returns:
            Sharpe ratio = (mean(returns)) / std(returns)
        """
        returns = np.array(returns)
        excess_returns = returns
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        return mean_return / (std_return + 1e-8)

    @staticmethod
    def construct_optimal_return_portfolio(returns, G = 5):
        """
        Given a 1D array of returns for one month, selects the top G assets (to long)
        and bottom G assets (to short) and computes weights proportional to their return magnitudes.
        
        Args:
            returns: 1D numpy array of monthly returns.
            G: Number of assets to long.
            
        Returns:
            weights: A numpy array of weights where long positions sum to +1,
                     short positions sum to -1, and unselected assets get 0.
        """
        returns = np.array(returns)
        n_assets = len(returns)
        if n_assets < 2 * G:
            raise ValueError("Not enough assets to select top G and bottom G positions.")
        
        sorted_idx = np.argsort(returns)[::-1]
        long_idx = sorted_idx[:G]
        short_idx = sorted_idx[-G:]
        
        long_returns = returns[long_idx]
        long_sum = long_returns.sum()
        if np.abs(long_sum) < 1e-8:
            long_weights = np.ones(G) / G
        else:
            long_weights = long_returns / long_sum
        
        short_returns = returns[short_idx]
        abs_short = np.abs(short_returns)
        short_sum = abs_short.sum()
        if np.abs(short_sum) < 1e-8:
            short_weights = -np.ones(G) / G
        else:
            short_weights = - (abs_short / short_sum)
        
        weights = np.zeros(n_assets)
        weights[long_idx] = long_weights
        weights[short_idx] = short_weights
        
        return weights

    def evaluate_baseline(self):
        """
        Evaluates a baseline strategy on the dataset.
        
        For each episode in the dataset (each episode contains T time steps with forward returns
        and corresponding masks):
          - For each month (time step), the function uses only valid asset data (mask==True)
            to select the top G and bottom G assets (weighted proportionally by return magnitude).
          - Computes the portfolio return for that month.
        
        Finally, computes the overall Sharpe ratio over all monthly portfolio returns.
        
        Returns:
            all_portfolio_returns: A numpy array of portfolio returns (one per month across all episodes).
            sharpe_ratio: Overall Sharpe ratio of the baseline portfolio returns.
        """
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        all_portfolio_returns = []
        
        # Loop over episodes.
        for episode_idx, (_, fwd_seq, mask_seq) in enumerate(dataloader):
            # fwd_seq and mask_seq have shape (1, T, num_assets)
            fwd_seq = fwd_seq.squeeze(0).numpy()   # shape: (T, num_assets)
            mask_seq = mask_seq.squeeze(0).numpy() # shape: (T, num_assets)
            T = fwd_seq.shape[0]
            for t in range(T):
                month_returns = fwd_seq[t, :]
                month_mask = mask_seq[t, :]
                # Use only assets with complete data.
                valid_returns = month_returns[month_mask]
                if len(valid_returns) < 2 * self.G:
                    port_return = 0.0
                else:
                    weights_valid = self.construct_optimal_return_portfolio(valid_returns, self.G)
                    # Map these weights back to full asset vector.
                    weights_full = np.zeros_like(month_returns)
                    valid_idx = np.where(month_mask)[0]
                    weights_full[valid_idx] = weights_valid
                    port_return = np.dot(weights_full, month_returns)
                all_portfolio_returns.append(port_return)
            logger.info(f"Processed episode {episode_idx+1} with {T} months.")
        
        all_portfolio_returns = np.array(all_portfolio_returns)
        overall_sharpe = self.compute_sharpe_ratio(all_portfolio_returns)
        logger.info(f"Overall baseline Sharpe ratio across all months: {overall_sharpe:.4f}")
        return all_portfolio_returns, overall_sharpe

# Example usage:
# Assuming you have an instance of your dataset and a trained model:
# dataset = AlphaPortfolioData(start_year=2014, end_year=2020, final_year=2016, lookback=12, T=12)
# model = YourTrainedAlphaPortfolioModel(...)
# evaluator = AlphaPortfolioEvaluator(dataset, model, device='cuda', G=5)
# model_sharpes = evaluator.test_model()         # Tests your deep RL model
# baseline_returns, baseline_sharpe = evaluator.evaluate_baseline()  # Baseline evaluation
