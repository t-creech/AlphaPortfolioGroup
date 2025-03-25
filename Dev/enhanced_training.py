from datetime import datetime
import os
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn # type: ignore
import numpy as np
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
from data import profile_function
from enhanced_visualization import VisualizationManager
import matplotlib.pyplot as plt
import torch.optim as optim
import seaborn as sns

def calculate_portfolio_std(
    weights: torch.Tensor, 
    future_returns: torch.Tensor,
    masks: Optional[torch.Tensor] = None, 
    return_step_values: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Calculate portfolio standard deviation from weights and future asset returns.
    
    Args:
        weights: Portfolio weights [batch_size, T, num_assets]
        future_returns: Future asset returns [batch_size, T, num_assets]
        masks: Optional masks for valid assets [batch_size, T, num_assets]
        return_step_values: Whether to return individual step values
        
    Returns:
        Portfolio standard deviation [batch_size] or
        Tuple of (std_deviation, step_returns)
    """
    # Calculate returns for each time step
    step_returns = calculate_returns(weights, future_returns, masks)  # [batch_size, T]
    
    # Calculate standard deviation across time steps
    std_dev = torch.std(step_returns, dim=1)  # [batch_size]
    
    if return_step_values:
        return std_dev, step_returns
    else:
        return std_dev

def calculate_sharpe_for_episode(
    returns: torch.Tensor,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 12.0
) -> torch.Tensor:
    """
    Calculate Sharpe ratio for an episode.
    
    Args:
        returns: Time series of returns [time_steps]
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns (12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return torch.tensor(0.0, device=returns.device)
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate Sharpe ratio
    mean_excess = torch.mean(excess_returns)
    std_excess = torch.std(excess_returns) + 1e-8  # Add small constant to avoid division by zero
    
    # Annualize Sharpe ratio
    sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(annualization_factor))
    
    return sharpe

def calculate_batch_sharpe(
    returns: torch.Tensor,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 12.0
) -> torch.Tensor:
    """
    Calculate Sharpe ratio for a batch of episodes.
    
    Args:
        returns: Time series of returns [batch_size, time_steps]
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns (12 for monthly)
        
    Returns:
        Sharpe ratios [batch_size]
    """
    batch_size, time_steps = returns.shape
    
    if time_steps < 2:
        return torch.zeros(batch_size, device=returns.device)
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate mean and std for each episode in the batch
    mean_excess = torch.mean(excess_returns, dim=1)  # [batch_size]
    std_excess = torch.std(excess_returns, dim=1) + 1e-8  # [batch_size]
    
    # Annualize Sharpe ratio
    sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(annualization_factor))
    
    return sharpe

def calculate_rolling_sharpe(
    returns: torch.Tensor,
    window_size: int = 12,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 12.0
) -> torch.Tensor:
    """
    Calculate rolling Sharpe ratio with a window.
    
    Args:
        returns: Time series of returns [batch_size, time_steps]
        window_size: Window size for rolling calculation
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns (12 for monthly)
        
    Returns:
        Rolling Sharpe ratios [batch_size, time_steps-window_size+1]
    """
    batch_size, time_steps = returns.shape
    
    if time_steps < window_size:
        return torch.zeros((batch_size, 1), device=returns.device)
    
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calculate rolling Sharpe ratios
    rolling_sharpes = []
    
    for t in range(window_size, time_steps + 1):
        # Get window of returns
        window_returns = returns[:, t-window_size:t]  # [batch_size, window_size]
        
        # Calculate excess returns
        excess_returns = window_returns - rf_period
        
        # Calculate mean and std for this window
        mean_excess = torch.mean(excess_returns, dim=1)  # [batch_size]
        std_excess = torch.std(excess_returns, dim=1) + 1e-8  # [batch_size]
        
        # Calculate Sharpe
        sharpe = (mean_excess / std_excess) * torch.sqrt(torch.tensor(annualization_factor))
        
        rolling_sharpes.append(sharpe)
    
    # Stack all windows
    if rolling_sharpes:
        rolling_sharpes = torch.stack(rolling_sharpes, dim=1)  # [batch_size, time_steps-window_size+1]
    else:
        rolling_sharpes = torch.zeros((batch_size, 1), device=returns.device)
    
    return rolling_sharpes

def calculate_portfolio_stats(
    weights: torch.Tensor,
    future_returns: torch.Tensor,
    masks: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Calculate comprehensive portfolio statistics.
    
    Args:
        weights: Portfolio weights [batch_size, T, num_assets]
        future_returns: Future asset returns [batch_size, T, num_assets]
        masks: Optional masks for valid assets [batch_size, T, num_assets]
        
    Returns:
        Dictionary with portfolio statistics
    """
    # Calculate returns
    returns = calculate_returns(weights, future_returns, masks)  # [batch_size, T]
    
    # Calculate various statistics
    mean_returns = torch.mean(returns, dim=1)  # [batch_size]
    std_returns = torch.std(returns, dim=1) + 1e-8  # [batch_size]
    sharpe_ratios = mean_returns / std_returns * torch.sqrt(torch.tensor(12.0))  # [batch_size]
    
    # Calculate cumulative returns
    cum_returns = torch.cumprod(1 + returns, dim=1) - 1  # [batch_size, T]
    final_cum_returns = cum_returns[:, -1]  # [batch_size]
    
    # Count positive returns
    positive_returns = torch.mean((returns > 0).float(), dim=1)  # [batch_size]
    
    # Calculate max drawdown
    peak_values = torch.cummax(cum_returns, dim=1)[0]  # [batch_size, T]
    drawdowns = (peak_values - cum_returns) / (peak_values + 1e-8)  # [batch_size, T]
    max_drawdowns = torch.max(drawdowns, dim=1)[0]  # [batch_size]
    
    # Calculate long/short exposures
    long_exposure = torch.sum(torch.clamp(weights, min=0), dim=2)  # [batch_size, T]
    short_exposure = torch.sum(torch.clamp(weights, max=0), dim=2)  # [batch_size, T]
    mean_long = torch.mean(long_exposure, dim=1)  # [batch_size]
    mean_short = torch.mean(-short_exposure, dim=1)  # [batch_size]
    
    # Calculate turnover (if T > 1)
    if weights.shape[1] > 1:
        weight_changes = torch.diff(weights, dim=1)  # [batch_size, T-1, num_assets]
        turnover = torch.sum(torch.abs(weight_changes), dim=(1, 2)) / (weights.shape[1] - 1)  # [batch_size]
    else:
        turnover = torch.zeros_like(mean_returns)  # [batch_size]
    
    # Return all statistics
    return {
        'returns': returns,  # [batch_size, T]
        'mean_returns': mean_returns,  # [batch_size]
        'std_returns': std_returns,  # [batch_size]
        'sharpe_ratios': sharpe_ratios,  # [batch_size]
        'cum_returns': cum_returns,  # [batch_size, T]
        'final_cum_returns': final_cum_returns,  # [batch_size]
        'positive_returns': positive_returns,  # [batch_size]
        'max_drawdowns': max_drawdowns,  # [batch_size]
        'mean_long_exposure': mean_long,  # [batch_size]
        'mean_short_exposure': mean_short,  # [batch_size]
        'turnover': turnover  # [batch_size]
    }

# -------------------------
# Utility Functions
# -------------------------
def plot_std_distribution(std_values: np.ndarray, output_dir: str, 
                        filename: str = "std_distribution.png", 
                        title: str = "Portfolio Standard Deviation Distribution"):
    """
    Plot distribution of portfolio standard deviations.
    
    Args:
        std_values: Standard deviation values
        output_dir: Output directory
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(std_values * 100, kde=True, bins=30)
    
    # Add mean line
    mean_std = np.mean(std_values) * 100
    plt.axvline(x=mean_std, color='red', linestyle='--', 
               label=f'Mean: {mean_std:.2f}%')
    
    # Add annotations
    plt.title(title, fontsize=14)
    plt.xlabel('Portfolio Standard Deviation (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = (
        f"Mean: {mean_std:.2f}%\n"
        f"Median: {np.median(std_values) * 100:.2f}%\n"
        f"Min: {np.min(std_values) * 100:.2f}%\n"
        f"Max: {np.max(std_values) * 100:.2f}%"
    )
    plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sharpe_distribution(sharpe_values: np.ndarray, output_dir: str, 
                           filename: str = "sharpe_distribution.png", 
                           title: str = "Sharpe Ratio Distribution"):
    """
    Plot distribution of Sharpe ratios.
    
    Args:
        sharpe_values: Sharpe ratio values
        output_dir: Output directory
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(sharpe_values, kde=True, bins=30)
    
    # Add mean line
    mean_sharpe = np.mean(sharpe_values)
    plt.axvline(x=mean_sharpe, color='red', linestyle='--', 
               label=f'Mean: {mean_sharpe:.4f}')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add annotations
    plt.title(title, fontsize=14)
    plt.xlabel('Sharpe Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = (
        f"Mean: {mean_sharpe:.4f}\n"
        f"Median: {np.median(sharpe_values):.4f}\n"
        f"Min: {np.min(sharpe_values):.4f}\n"
        f"Max: {np.max(sharpe_values):.4f}\n"
        f"Positive Sharpe: {np.mean(sharpe_values > 0) * 100:.1f}%"
    )
    plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sharpe_vs_std(sharpe_values: np.ndarray, std_values: np.ndarray, 
                     output_dir: str, filename: str = "sharpe_vs_std.png", 
                     title: str = "Sharpe Ratio vs Standard Deviation"):
    """
    Plot Sharpe ratio against standard deviation for risk-return analysis.
    
    Args:
        sharpe_values: Sharpe ratio values
        std_values: Standard deviation values
        output_dir: Output directory
        filename: Output filename
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with coloring by Sharpe
    sc = plt.scatter(std_values * 100, sharpe_values, c=sharpe_values, 
                   cmap='viridis', alpha=0.7, s=80)
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    
    # Add mean lines
    plt.axvline(x=np.mean(std_values) * 100, color='r', linestyle='--', alpha=0.5, 
               label=f'Mean Std Dev: {np.mean(std_values)*100:.2f}%')
    plt.axhline(y=np.mean(sharpe_values), color='g', linestyle='--', alpha=0.5, 
               label=f'Mean Sharpe: {np.mean(sharpe_values):.4f}')
    
    # Add horizontal line at Sharpe = 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Portfolio Standard Deviation (%)', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Optimized function to calculate returns
def calculate_returns(weights: torch.Tensor, future_returns: torch.Tensor, 
                    masks: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate portfolio returns efficiently from weights and future asset returns.
    
    Args:
        weights: Portfolio weights [batch_size, T, num_assets]
        future_returns: Future asset returns [batch_size, T, num_assets]
        masks: Optional masks for valid assets [batch_size, T, num_assets]
        
    Returns:
        Portfolio returns [batch_size, T]
    """
    # Apply masks if provided (do this once)
    if masks is not None:
        masked_weights = weights * masks
    else:
        masked_weights = weights
    
    # Calculate portfolio returns in one operation
    # This is more efficient than looping through time steps
    portfolio_returns = torch.sum(masked_weights * future_returns, dim=-1)  # [batch_size, T]
    
    return portfolio_returns


class RLTrainer:
    """
    Reinforcement Learning trainer for AlphaPortfolio.
    Handles training, validation, tracking and visualization of metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 0.5,
        sharpe_window: int = 12,
        gamma: float = 0.99,
        device: torch.device = None
    ):
        """
        Initialize RL trainer.
        
        Args:
            model: AlphaPortfolio model
            lr: Learning rate
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for gradient clipping
            sharpe_window: Window size for Sharpe calculation
            gamma: Discount factor for RL
            device: Device to train on
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.sharpe_window = sharpe_window
        self.gamma = gamma
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # For tracking episode-level metrics
        self.episode_returns = []
        self.episode_stds = []
        self.episode_sharpes = []
        
        # For tracking batch-level metrics
        self.batch_metrics = []
        
        logging.info(f"Initialized RL trainer with lr={lr}, weight_decay={weight_decay}, device={device}")
    
    @profile_function
    def train_epoch(self, train_loader, epoch: int, output_dir: str, cycle_idx: int, 
               param_set_id: Optional[str] = None) -> Dict[str, float]:
        """Train for one epoch with parameter set specific directories."""
        self.model.train()
        metrics_list = []
        
        # Create parameter-specific output directory
        if param_set_id:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        else:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        
        os.makedirs(epoch_dir, exist_ok=True)

        epoch_returns = []
        epoch_weights = []
        epoch_scores = []
        epoch_indices = []
        epoch_stds = []
        epoch_sharpes = []
        
        # Reset episode-level trackers
        self.episode_returns = []
        self.episode_stds = []
        self.episode_sharpes = []
        
        # Initialize scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        try:
            logging.info(f"Starting training for epoch {epoch} with {len(train_loader)} batches")
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
                # Safely unpack the batch
                # logging.info(f"batch index {batch_idx}, batch {batch}", "len batch", {len(batch)})
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    states, future_returns, masks = batch
                else:
                    logging.error(f"Unexpected batch format: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                    continue
                    
                states = states.to(self.device)
                future_returns = future_returns.to(self.device)
                masks = masks.to(self.device)
                
                # logging.info(f"states {states.shape}, future_returns {future_returns.shape}, masks {masks.shape}")
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                        
                        # Calculate portfolio returns
                        batch_size, T, num_assets = future_returns.shape
                        portfolio_returns_timestep = calculate_returns(
                            portfolio_weights,
                            future_returns,
                            masks
                        )  # [batch_size, T]
                        
                        # Calculate portfolio standard deviation
                        portfolio_stds = torch.std(portfolio_returns_timestep, dim=1) + 1e-8  # [batch_size]
                        
                        # Calculate mean portfolio returns
                        mean_portfolio_returns = torch.mean(portfolio_returns_timestep, dim=1)  # [batch_size]
                        
                        # Calculate Sharpe ratio for each episode
                        sharpe_ratios = (mean_portfolio_returns / portfolio_stds) * torch.sqrt(torch.tensor(12.0))
                        
                        # Use mean Sharpe ratio as reward
                        mean_sharpe = torch.mean(sharpe_ratios)
                        
                        # Loss is negative Sharpe ratio (we want to maximize Sharpe)
                        loss = -mean_sharpe
                    
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Standard forward pass without mixed precision
                    portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                    
                    # Calculate portfolio returns
                    batch_size, T, num_assets = future_returns.shape
                    portfolio_returns_timestep = calculate_returns(
                        portfolio_weights,
                        future_returns,
                        masks
                    )  # [batch_size, T]
                    
                    # Calculate portfolio standard deviation
                    portfolio_stds = torch.std(portfolio_returns_timestep, dim=1) + 1e-8  # [batch_size]
                    
                    # Calculate mean portfolio returns
                    mean_portfolio_returns = torch.mean(portfolio_returns_timestep, dim=1)  # [batch_size]
                    
                    # Calculate Sharpe ratio for each episode
                    sharpe_ratios = (mean_portfolio_returns / portfolio_stds) * torch.sqrt(torch.tensor(12.0))
                    
                    # Use mean Sharpe ratio as reward
                    mean_sharpe = torch.mean(sharpe_ratios)
                    
                    # Loss is negative Sharpe ratio (we want to maximize Sharpe)
                    loss = -mean_sharpe
                    
                    # Regular backward pass
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Track for epoch-level metrics
                epoch_returns.extend(portfolio_returns_timestep.detach().cpu().numpy())
                epoch_stds.extend(portfolio_stds.detach().cpu().numpy())
                epoch_sharpes.extend(sharpe_ratios.detach().cpu().numpy())
                
                # Store episode metrics for detailed tracking
                for b in range(batch_size):
                    # Track individual episode metrics
                    # logging.info(f"batch index {batch_idx}, portfolio_returns_timestep {portfolio_returns_timestep[b].detach().cpu().numpy()}")
                    self.episode_returns.append(portfolio_returns_timestep[b].detach().cpu().numpy())
                    self.episode_stds.append(portfolio_stds[b].item())
                    self.episode_sharpes.append(sharpe_ratios[b].item())
                
                # Calculate batch metrics
                metrics = {
                    'loss': loss.item(),
                    'sharpe_ratio': mean_sharpe.item(),
                    'mean_return': mean_portfolio_returns.mean().item(),
                    'std_return': portfolio_stds.mean().item(),
                    'mean_weight_long': portfolio_weights[portfolio_weights > 0].mean().item() if (portfolio_weights > 0).any() else 0,
                    'mean_weight_short': portfolio_weights[portfolio_weights < 0].mean().item() if (portfolio_weights < 0).any() else 0
                }
                
                metrics_list.append(metrics)
                
                episode_dir = os.path.join(epoch_dir, "episodes")
                os.makedirs(episode_dir, exist_ok=True)
                
                # # Visualize episodes 
                # if batch_idx == 0:
                    # Visualize up to 3 episodes from first batch
                for ep_idx in range(batch_size):
                    self._visualize_detailed_portfolio(
                                    episode_idx=ep_idx,
                                    weights=portfolio_weights[ep_idx].detach().cpu().numpy(),
                                    returns=portfolio_returns_timestep[ep_idx].detach().cpu().numpy(),
                                    winner_scores=winner_scores[ep_idx].detach().cpu().numpy(),
                                    output_dir=episode_dir,
                                    epoch=epoch,
                                    batch_idx=batch_idx,
                                    phase="train"
                                )
                
                # Log batch progress sparingly to avoid slowdown
                if batch_idx % 5 == 0:
                    metrics_avg = {k: np.mean([m[k] for m in metrics_list[-5:]]) for k in metrics_list[-1].keys()}
                    logging.info(f"  Batch {batch_idx}/{len(train_loader)}: {metrics_avg}")
            
            # Calculate average metrics
            metrics_avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys() if metrics_list}
            logging.info(f"Epoch {epoch} training complete: {metrics_avg}")
            
            # Generate visualizations for standard deviation and Sharpe
            if len(self.episode_returns) > 0:
                # Generate visualizations in separate directory
                viz_dir = os.path.join(output_dir, f"cycle_{cycle_idx}", f"epoch_{epoch}")
                os.makedirs(viz_dir, exist_ok=True)
                
                # Plot standard deviation distribution
                plot_std_distribution(
                    np.array(self.episode_stds),
                    output_dir=viz_dir,
                    filename=f"std_distribution.png",
                    title=f"Portfolio Standard Deviation Distribution (Epoch {epoch})"
                )
                
                # Plot Sharpe ratio distribution
                plot_sharpe_distribution(
                    np.array(self.episode_sharpes),
                    output_dir=viz_dir,
                    filename=f"sharpe_distribution.png",
                    title=f"Sharpe Ratio Distribution (Epoch {epoch})"
                )
                
                # Plot Sharpe vs standard deviation scatter
                plot_sharpe_vs_std(
                    np.array(self.episode_sharpes),
                    np.array(self.episode_stds),
                    output_dir=viz_dir,
                    filename=f"sharpe_vs_std.png",
                    title=f"Sharpe vs Standard Deviation (Epoch {epoch})"
                )
        
            return metrics_avg
            
        except Exception as e:
            logging.error(f"Error in train_epoch: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")
            # Return empty metrics in case of error
            return {'loss': float('inf'), 'sharpe_ratio': -float('inf'), 'mean_return': 0.0, 'std_return': 0.0}

    def _visualize_detailed_portfolio(self, episode_idx, weights, returns, winner_scores, output_dir, 
                               epoch=None, batch_idx=None, phase=None):
        """
        Create detailed portfolio visualizations for a single episode.
        
        Args:
            episode_idx: Episode index or identifier
            weights: Portfolio weights for this episode
            returns: Returns for this episode
            winner_scores: Winner scores for this episode
            output_dir: Directory to save visualizations
            epoch: Optional epoch number for the filename (backwards compatibility)
            batch_idx: Optional batch index for the filename (backwards compatibility)
            phase: Optional phase name for the filename (backwards compatibility)
        """
        # Create subfolder for this episode
        episode_folder_name = f"episode_{episode_idx}"
        
        # Add epoch and batch info to folder name if provided (for compatibility)
        if epoch is not None and batch_idx is not None:
            episode_folder_name = f"episode_{episode_idx}_epoch_{epoch}_batch_{batch_idx}"
        elif epoch is not None:
            episode_folder_name = f"episode_{episode_idx}_epoch_{epoch}"
            
        if phase is not None:
            episode_folder_name = f"{phase}_{episode_folder_name}"
            
        episode_dir = os.path.join(output_dir, episode_folder_name)
        os.makedirs(episode_dir, exist_ok=True)
        
        # Title construction
        base_title = f"Episode {episode_idx}"
        if epoch is not None:
            base_title += f" (Epoch {epoch}"
            if batch_idx is not None:
                base_title += f", Batch {batch_idx}"
            base_title += ")"
        
        # 1. Plot returns by time step
        self._plot_returns_by_timestep(
            returns=returns,
            output_dir=episode_dir,
            filename="returns.png",
            title=f"{base_title} Returns by Time Step"
        )
        
        # 2. Plot cumulative returns
        self._plot_cumulative_returns(
            returns=returns,
            output_dir=episode_dir,
            filename="cumulative_returns.png",
            title=f"{base_title} Cumulative Returns"
        )
        
        # 3. Plot weights at each time step
        for t in range(len(weights)):
            self._plot_weights_at_timestep(
                weights=weights[t],
                output_dir=episode_dir,
                filename=f"weights_t{t+1}.png",
                title=f"{base_title} Portfolio Weights at t={t+1}"
            )
        
        # 4. Plot portfolio allocation over time (heatmap)
        self._plot_portfolio_allocation_over_time(
            weights=weights,
            output_dir=episode_dir,
            filename="portfolio_allocation.png",
            title=f"{base_title} Portfolio Allocation Over Time"
        )
    
    def _visualize_aggregate_portfolio(self, all_returns, all_weights, output_dir):
        """Create aggregate portfolio visualizations across all episodes."""
        # 1. Plot distribution of returns
        plt.figure(figsize=(12, 8))
        sns.histplot(all_returns.flatten() * 100, kde=True, bins=30)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=np.mean(all_returns.flatten()) * 100, color='blue', linestyle='-', 
                label=f'Mean: {np.mean(all_returns.flatten()) * 100:.2f}%')
        
        plt.title('Distribution of Portfolio Returns', fontsize=14)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        stats_text = (
            f"Mean: {np.mean(all_returns.flatten()) * 100:.2f}%\n"
            f"Median: {np.median(all_returns.flatten()) * 100:.2f}%\n"
            f"Std Dev: {np.std(all_returns.flatten()) * 100:.2f}%\n"
            f"Min: {np.min(all_returns.flatten()) * 100:.2f}%\n"
            f"Max: {np.max(all_returns.flatten()) * 100:.2f}%\n"
            f"Positive Returns: {np.mean(all_returns.flatten() > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, "returns_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot distribution of weights
        plt.figure(figsize=(12, 8))
        sns.histplot(all_weights.flatten(), kde=True, bins=30)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Distribution of Portfolio Weights', fontsize=14)
        plt.xlabel('Weight', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        stats_text = (
            f"Mean: {np.mean(all_weights.flatten()):.4f}\n"
            f"Median: {np.median(all_weights.flatten()):.4f}\n"
            f"Std Dev: {np.std(all_weights.flatten()):.4f}\n"
            f"Min: {np.min(all_weights.flatten()):.4f}\n"
            f"Max: {np.max(all_weights.flatten()):.4f}\n"
            f"Long Weights: {np.mean(all_weights.flatten() > 0) * 100:.1f}%\n"
            f"Short Weights: {np.mean(all_weights.flatten() < 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, "weights_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot average cumulative returns across all episodes
        cum_returns = np.cumprod(1 + all_returns, axis=1) - 1
        mean_cum_returns = np.mean(cum_returns, axis=0)
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(mean_cum_returns) + 1), mean_cum_returns * 100, 'b-', linewidth=2)
        plt.fill_between(
            range(1, len(mean_cum_returns) + 1),
            (mean_cum_returns - np.std(cum_returns, axis=0)) * 100,
            (mean_cum_returns + np.std(cum_returns, axis=0)) * 100,
            alpha=0.2,
            color='blue'
        )
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Average Cumulative Portfolio Returns', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "average_cumulative_returns.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_portfolio_distributions(self, all_returns, all_weights, output_dir):
        """Visualize distributions of portfolio metrics."""
        # Calculate Sharpe ratio for each episode
        episode_sharpes = []
        episode_returns = []
        episode_stds = []
        
        for i in range(all_returns.shape[0]):
            returns = all_returns[i]
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8
            sharpe = mean_return / std_return * np.sqrt(12)  # Annualized
            
            episode_sharpes.append(sharpe)
            episode_returns.append(mean_return)
            episode_stds.append(std_return)
        
        # 1. Plot Sharpe ratio distribution
        plt.figure(figsize=(12, 8))
        sns.histplot(episode_sharpes, kde=True, bins=30)
        plt.axvline(x=np.mean(episode_sharpes), color='blue', linestyle='-', 
                label=f'Mean: {np.mean(episode_sharpes):.4f}')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.title('Distribution of Episode Sharpe Ratios', fontsize=14)
        plt.xlabel('Sharpe Ratio', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        stats_text = (
            f"Mean: {np.mean(episode_sharpes):.4f}\n"
            f"Median: {np.median(episode_sharpes):.4f}\n"
            f"Std Dev: {np.std(episode_sharpes):.4f}\n"
            f"Min: {np.min(episode_sharpes):.4f}\n"
            f"Max: {np.max(episode_sharpes):.4f}\n"
            f"Positive Sharpe: {np.mean(np.array(episode_sharpes) > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, "sharpe_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot Sharpe vs Standard Deviation scatter
        plt.figure(figsize=(12, 8))
        plt.scatter(np.array(episode_stds) * 100, episode_sharpes, c=episode_sharpes, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Sharpe Ratio')
        
        plt.title('Sharpe Ratio vs. Standard Deviation', fontsize=14)
        plt.xlabel('Standard Deviation (%)', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "sharpe_vs_std.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot average portfolio positioning
        long_counts = np.mean(np.sum(all_weights > 0, axis=2), axis=0)
        short_counts = np.mean(np.sum(all_weights < 0, axis=2), axis=0)
        
        plt.figure(figsize=(12, 8))
        x = range(1, len(long_counts) + 1)
        
        plt.bar(x, long_counts, color='green', alpha=0.7, label='Long Positions')
        plt.bar(x, -short_counts, color='red', alpha=0.7, label='Short Positions')
        
        plt.title('Average Number of Positions by Time Step', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Number of Positions', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "average_positions.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_weights_at_timestep(self, weights: np.ndarray, output_dir: str, 
                                filename: str, title: str):
        """Plot portfolio weights at a specific time step."""
        plt.figure(figsize=(12, 10))
        
        # Sort weights by magnitude
        sorted_indices = np.argsort(-np.abs(weights))
        top_indices = sorted_indices[:min(20, len(weights))]
        
        # Get asset names
        if hasattr(self, 'assets_list') and len(self.assets_list) >= len(weights):
            top_assets = [self.assets_list[i] for i in top_indices]
        else:
            top_assets = [f"Asset {i}" for i in top_indices]
        
        # Create colors based on sign
        colors = ['green' if w > 0 else 'red' for w in weights[top_indices]]
        
        # Create horizontal bar chart
        plt.barh(top_assets, weights[top_indices], color=colors)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add annotations with exact weight values
        for i, w in enumerate(weights[top_indices]):
            plt.text(w + (0.001 if w >= 0 else -0.005), i, f'{w:.4f}', 
                    ha='left' if w >= 0 else 'right', va='center')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Weight', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add summary statistics
        long_count = np.sum(weights > 0)
        short_count = np.sum(weights < 0)
        long_allocation = np.sum(weights[weights > 0])
        short_allocation = np.sum(weights[weights < 0])
        
        stats_text = (
            f"Long positions: {long_count}\n"
            f"Short positions: {short_count}\n"
            f"Long allocation: {long_allocation:.4f}\n"
            f"Short allocation: {short_allocation:.4f}"
        )
        
        plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_returns_by_timestep(self, returns: np.ndarray, output_dir: str, 
                                filename: str, title: str):
        """Plot detailed returns at each time step."""
        T = len(returns)
        plt.figure(figsize=(12, 6))
        
        # Create bar chart with colored bars based on sign
        colors = ['green' if r > 0 else 'red' for r in returns]
        plt.bar(range(1, T+1), returns * 100, color=colors)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add return values as text
        for i, r in enumerate(returns):
            plt.text(i+1, r*100 + (0.5 if r >= 0 else -1.5), f'{r*100:.2f}%', 
                    ha='center', va='bottom' if r >= 0 else 'top')
        
        # Add mean return line
        mean_return = np.mean(returns) * 100
        plt.axhline(y=mean_return, color='blue', linestyle='-', 
                   alpha=0.7, label=f'Mean: {mean_return:.2f}%')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(range(1, T+1))
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cumulative_returns(self, returns: np.ndarray, output_dir: str, 
                               filename: str, title: str):
        """Plot cumulative returns over time steps."""
        if len(returns) == 0:
            return
            
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        
        plt.figure(figsize=(12, 6))
        
        # Plot cumulative returns
        plt.plot(range(1, len(returns) + 1), cum_returns * 100, 'b-o', linewidth=2)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add final return annotation
        final_return = cum_returns[-1] * 100
        plt.annotate(f"Final: {final_return:.2f}%", 
                    xy=(len(returns), final_return), 
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center")
        
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(returns) + 1))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_allocation_over_time(self, weights: np.ndarray, output_dir: str, 
                                          filename: str, title: str):
        """Plot portfolio allocation over time with detailed visualizations."""
        if len(weights) == 0:
            return
            
        plt.figure(figsize=(14, 10))
        
        # Create a GridSpec for multiple subplots
        gs = GridSpec(2, 2, figure=plt.gcf())
        ax1 = plt.subplot(gs[0, :])  # Top panel for heatmap
        ax2 = plt.subplot(gs[1, 0])  # Bottom left for long positions
        ax3 = plt.subplot(gs[1, 1])  # Bottom right for short positions
        
        # Select top assets by absolute weight for visualization
        mean_abs_weights = np.mean(np.abs(weights), axis=0)
        top_indices = np.argsort(-mean_abs_weights)[:15]
        
        # Get asset names
        if hasattr(self, 'assets_list') and len(self.assets_list) >= weights.shape[1]:
            top_assets = [self.assets_list[i] for i in top_indices]
        else:
            top_assets = [f"Asset {i}" for i in top_indices]
        
        # Create heatmap in top panel
        selected_weights = weights[:, top_indices]
        im = sns.heatmap(
            selected_weights.T,  # Transpose to have assets as rows
            cmap='RdBu_r',
            center=0,
            vmin=-np.max(np.abs(selected_weights)),
            vmax=np.max(np.abs(selected_weights)),
            yticklabels=top_assets,
            xticklabels=[f"t={i+1}" for i in range(len(weights))],
            cbar_kws={'label': 'Portfolio Weight'},
            ax=ax1
        )
        
        # Make sure all labels are visible
        plt.setp(ax1.get_yticklabels(), fontsize=8)
        
        ax1.set_title('Portfolio Allocation Over Time (Top 15 Assets)', fontsize=12)
        ax1.set_xlabel('Time Step', fontsize=10)
        ax1.set_ylabel('Asset', fontsize=10)
        
        # Calculate position counts for each time step
        long_counts = np.sum(weights > 0, axis=1)
        short_counts = np.sum(weights < 0, axis=1)
        
        # Plot position counts over time
        ax2.bar(range(1, len(weights) + 1), long_counts, color='green', alpha=0.7, label='Long')
        ax2.bar(range(1, len(weights) + 1), -short_counts, color='red', alpha=0.7, label='Short')
        ax2.set_title('Number of Positions by Time Step', fontsize=12)
        ax2.set_xlabel('Time Step', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(range(1, len(weights) + 1))
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Calculate allocation for each time step
        long_allocation = np.sum(np.maximum(weights, 0), axis=1)
        short_allocation = np.sum(np.minimum(weights, 0), axis=1)
        
        # Plot allocation over time
        ax3.bar(range(1, len(weights) + 1), long_allocation, color='green', alpha=0.7, label='Long')
        ax3.bar(range(1, len(weights) + 1), -short_allocation, color='red', alpha=0.7, label='Short')
        ax3.set_title('Allocation by Time Step', fontsize=12)
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Allocation', fontsize=10)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticks(range(1, len(weights) + 1))
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    """
    Comprehensive training manager for AlphaPortfolio.
    Handles training, validation, visualization, and logging.
    """
    
    def __init__(self, 
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               config: Any,
               cycle_params: Dict[str, Any],
               device: torch.device,
               sharpe_window: int = 12,
               max_grad_norm: float = 0.5):
        """
        Initialize training manager.
        
        Args:
            model: AlphaPortfolio model
            optimizer: Optimizer
            config: Configuration object
            cycle_params: Parameters for current cycle
            device: Device to train on
            sharpe_window: Window size for Sharpe calculation
            max_grad_norm: Maximum gradient norm for gradient clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.cycle_params = cycle_params
        self.device = device
        self.sharpe_window = sharpe_window
        self.max_grad_norm = max_grad_norm
        
        # Extract parameters
        self.cycle_idx = cycle_params["cycle_idx"]
        self.output_dir = config.config["paths"]["plot_dir"]
        self.model_dir = config.config["paths"]["model_dir"]
        self.T = config.config["model"]["T"]
        self.num_epochs = config.config["training"]["num_epochs"]
        self.early_stopping_patience = config.config["training"]["patience"]
        
        # Create visualization manager
        self.vis_manager = None  # Will be initialized later with assets_list
        
        # Initialize trackers
        self.episode_tracker = EpisodeTracker(T=self.T, num_assets=0)  # Will be updated with actual num_assets
        self.train_batch_tracker = BatchTracker()
        self.val_batch_tracker = BatchTracker()
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'train_sharpe': [],
            'train_returns': [],
            'train_std': [],
            'val_loss': [],
            'val_sharpe': [],
            'val_returns': [],
            'val_std': [],
            'mean_weight_long': [],
            'mean_weight_short': []
        }
        
        # Initialize early stopping
        self.best_val_sharpe = -float('inf')
        self.early_stopping_counter = 0
        self.early_stop = False
        
        logging.info(f"Initialized training manager for cycle {self.cycle_idx}")
    
    def initialize_visualization(self, num_assets: int, assets_list: Optional[List[str]] = None):
        """
        Initialize visualization manager.
        
        Args:
            num_assets: Number of assets in the universe
            assets_list: Optional list of asset names
        """
        self.episode_tracker = EpisodeTracker(T=self.T, num_assets=num_assets)
        self.vis_manager = VisualizationManager(
            output_dir=self.output_dir,
            cycle_idx=self.cycle_idx,
            T=self.T,
            num_assets=num_assets,
            assets_list=assets_list
        )
    
    def train_full_epochs(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """
        Train for all epochs before validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with training metrics
        """
        logging.info(f"Starting training for {self.num_epochs} epochs")
        
        # Initialize visualization if needed
        if self.vis_manager is None:
            num_assets = train_loader.dataset.global_max_assets
            assets_list = None
            if hasattr(train_loader.dataset, 'get_asset_names'):
                assets_list = train_loader.dataset.get_asset_names()
            self.initialize_visualization(num_assets, assets_list)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update metrics
            self.metrics['train_loss'].append(train_metrics.get('loss', float('inf')))
            self.metrics['train_sharpe'].append(train_metrics.get('sharpe_ratio', -float('inf')))
            self.metrics['train_returns'].append(train_metrics.get('mean_return', 0.0))
            self.metrics['train_std'].append(train_metrics.get('std_return', 0.0))
            self.metrics['val_loss'].append(val_metrics.get('loss', float('inf')))
            self.metrics['val_sharpe'].append(val_metrics.get('sharpe_ratio', -float('inf')))
            self.metrics['val_returns'].append(val_metrics.get('mean_return', 0.0))
            self.metrics['val_std'].append(val_metrics.get('std_return', 0.0))
            self.metrics['mean_weight_long'].append(train_metrics.get('mean_weight_long', 0.0))
            self.metrics['mean_weight_short'].append(train_metrics.get('mean_weight_short', 0.0))
            
            # Visualize training progress
            self.vis_manager.visualize_training_progress(
                metrics=self.metrics,
                output_filename=f"cycle_{self.cycle_idx}_training_progress.png"
            )
            
            # Save model if validation Sharpe improved
            if val_metrics.get('sharpe_ratio', -float('inf')) > self.best_val_sharpe:
                self.best_val_sharpe = val_metrics.get('sharpe_ratio', -float('inf'))
                
                # Reset early stopping counter
                self.early_stopping_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_sharpe': self.best_val_sharpe,
                    'metrics': self.metrics,
                    'hyperparameters': {
                        'd_model': self.config.config["model"]["d_model"],
                        'nhead': self.config.config["model"]["nhead"],
                        'num_layers': self.config.config["model"]["num_layers"],
                        'G': self.config.config["model"]["G"],
                        'learning_rate': self.config.config["training"]["learning_rate"],
                        'weight_decay': self.config.config["training"]["weight_decay"]
                    }
                }, os.path.join(self.model_dir, f"cycle_{self.cycle_idx}_best.pt"))
                
                logging.info(f"  New best model saved with validation Sharpe: {self.best_val_sharpe:.4f}")
            else:
                # Increment early stopping counter
                self.early_stopping_counter += 1
                logging.info(f"  Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.early_stop = True
                    logging.info(f"  Early stopping triggered after epoch {epoch}")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_sharpe': val_metrics.get('sharpe_ratio', -float('inf')),
                    'metrics': self.metrics
                }, os.path.join(self.model_dir, f"cycle_{self.cycle_idx}_epoch_{epoch}.pt"))
            
            # Log epoch summary
            epoch_time = time.time() - start_time
            logging.info(f"  Epoch {epoch} completed in {epoch_time:.2f}s, "
                       f"Train Sharpe: {train_metrics.get('sharpe_ratio', -float('inf')):.4f}, "
                       f"Val Sharpe: {val_metrics.get('sharpe_ratio', -float('inf')):.4f}")
            
            # Check early stopping
            if self.early_stop:
                break
        
        # Load best model
        best_checkpoint_path = os.path.join(self.model_dir, f"cycle_{self.cycle_idx}_best.pt")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model from epoch {checkpoint['epoch']} with validation Sharpe: {checkpoint['val_sharpe']:.4f}")
        
        logging.info(f"Training completed for cycle {self.cycle_idx}")
        
        return self.metrics
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_batch_tracker.reset()
        
        try:
            logging.info(f"Starting training for epoch {epoch} with {len(train_loader)} batches")
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
                # Safely unpack the batch
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    states, future_returns, masks = batch
                else:
                    logging.error(f"Unexpected batch format: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                    continue
                    
                states = states.to(self.device)
                future_returns = future_returns.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                try:
                    logging.debug(f"Performing forward pass for batch {batch_idx}")
                    portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                    logging.debug(f"Forward pass completed for batch {batch_idx}")
                except Exception as e:
                    logging.error(f"Error in forward pass: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise
                
                # Calculate portfolio returns over time
                try:
                    batch_size, T, num_assets = future_returns.shape
                    logging.debug(f"Processing batch with shape: [{batch_size}, {T}, {num_assets}]")
                    
                    # Calculate returns for each time step
                    portfolio_returns_timestep = self.calculate_returns(
                        portfolio_weights,
                        future_returns,
                        masks
                    )  # [batch_size, T]
                    
                    # Calculate portfolio standard deviation
                    portfolio_stds = torch.std(portfolio_returns_timestep, dim=1)  # [batch_size]
                    
                    # Calculate mean portfolio return
                    mean_portfolio_returns = torch.mean(portfolio_returns_timestep, dim=1)  # [batch_size]
                    
                    # Calculate Sharpe ratio for each batch element
                    sharpe_ratios = mean_portfolio_returns / (portfolio_stds + 1e-8)  # Add small constant to avoid division by zero
                    
                    # Scale to annualized Sharpe (assuming monthly returns)
                    annualization_factor = 12.0
                    sharpe_ratios = sharpe_ratios * torch.sqrt(torch.tensor(annualization_factor))
                    
                    # Mean Sharpe ratio across the batch
                    mean_sharpe = torch.mean(sharpe_ratios)
                    
                    # Loss is negative Sharpe ratio (we want to maximize Sharpe)
                    loss = -mean_sharpe
                    
                except Exception as e:
                    logging.error(f"Error calculating returns and Sharpe ratio: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise
                
                # Backward pass and optimization
                try:
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Update model
                    self.optimizer.step()
                except Exception as e:
                    logging.error(f"Error in backward pass or optimization: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise
                
                # Calculate batch metrics
                batch_metrics = {
                    'loss': loss.item(),
                    'sharpe_ratio': mean_sharpe.item(),
                    'mean_return': mean_portfolio_returns.mean().item(),
                    'std_return': portfolio_stds.mean().item(),
                    'mean_weight_long': portfolio_weights[portfolio_weights > 0].mean().item() if (portfolio_weights > 0).any() else 0,
                    'mean_weight_short': portfolio_weights[portfolio_weights < 0].mean().item() if (portfolio_weights < 0).any() else 0
                }
                
                # Track batch metrics
                self.train_batch_tracker.add_batch(
                    batch_metrics=batch_metrics,
                    returns=portfolio_returns_timestep.detach().cpu().numpy(),
                    weights=portfolio_weights.detach().cpu().numpy(),
                    winner_scores=winner_scores.detach().cpu().numpy(),
                    sharpe_ratios=sharpe_ratios.detach().cpu().numpy()
                )
                
                # Visualize episodes from first batch
                if batch_idx == 0:
                    for ep_idx in range(min(5, batch_size)):  # Visualize up to 5 episodes
                        self.visualize_episode(
                            episode_idx=ep_idx,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            phase='train',
                            returns=portfolio_returns_timestep[ep_idx].detach().cpu().numpy(),
                            weights=portfolio_weights[ep_idx].detach().cpu().numpy(),
                            winner_scores=winner_scores[ep_idx].detach().cpu().numpy()
                        )
                
                # Visualize batch
                if batch_idx % 5 == 0:  # Visualize every 5th batch
                    self.visualize_batch(
                        batch_idx=batch_idx,
                        epoch=epoch,
                        phase='train',
                        returns=portfolio_returns_timestep.detach().cpu().numpy(),
                        weights=portfolio_weights.detach().cpu().numpy(),
                        sharpe_ratios=sharpe_ratios.detach().cpu().numpy(),
                        winner_scores=winner_scores.detach().cpu().numpy()
                    )
                
                # Log batch progress
                if batch_idx % 5 == 0:
                    logging.info(f"  Batch {batch_idx}/{len(train_loader)}: Loss={batch_metrics['loss']:.4f}, Sharpe={batch_metrics['sharpe_ratio']:.4f}")
            
            # Get epoch metrics
            epoch_metrics = self.train_batch_tracker.get_epoch_metrics()
            
            # Visualize epoch
            all_returns = self.train_batch_tracker.get_all_returns()
            all_weights = self.train_batch_tracker.get_all_weights()
            all_sharpe_ratios = self.train_batch_tracker.get_all_sharpe_ratios()
            
            self.visualize_epoch(
                epoch=epoch,
                phase='train',
                all_returns=all_returns,
                all_weights=all_weights,
                all_sharpe_ratios=all_sharpe_ratios,
                metrics=epoch_metrics
            )
            
            return epoch_metrics
            
        except Exception as e:
            logging.error(f"Error in train_epoch: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")
            # Return empty metrics in case of error
            return {'loss': float('inf'), 'sharpe_ratio': -float('inf'), 'mean_return': 0.0}
    
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_batch_tracker.reset()
        
        try:
            logging.info(f"Starting validation for epoch {epoch}")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
                    # Safely unpack the batch
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        states, future_returns, masks = batch
                    else:
                        logging.error(f"Unexpected validation batch format: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                        continue
                    
                    states = states.to(self.device)
                    future_returns = future_returns.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Forward pass
                    try:
                        portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                    except Exception as e:
                        logging.error(f"Error in validation forward pass: {str(e)}")
                        logging.error(traceback.format_exc())
                        raise
                    
                    # Calculate portfolio returns
                    try:
                        # Calculate returns for each time step
                        portfolio_returns_timestep = self.calculate_returns(
                            portfolio_weights,
                            future_returns,
                            masks
                        )  # [batch_size, T]
                        
                        # Calculate portfolio standard deviation
                        portfolio_stds = torch.std(portfolio_returns_timestep, dim=1)  # [batch_size]
                        
                        # Calculate mean portfolio return
                        mean_portfolio_returns = torch.mean(portfolio_returns_timestep, dim=1)  # [batch_size]
                        
                        # Calculate Sharpe ratio for each batch element
                        sharpe_ratios = mean_portfolio_returns / (portfolio_stds + 1e-8)  # Add small constant to avoid division by zero
                        
                        # Scale to annualized Sharpe (assuming monthly returns)
                        annualization_factor = 12.0
                        sharpe_ratios = sharpe_ratios * torch.sqrt(torch.tensor(annualization_factor))
                        
                        # Mean Sharpe ratio across the batch
                        mean_sharpe = torch.mean(sharpe_ratios)
                        
                        # Loss is negative Sharpe ratio
                        loss = -mean_sharpe
                        
                    except Exception as e:
                        logging.error(f"Error calculating validation returns and metrics: {str(e)}")
                        logging.error(traceback.format_exc())
                        raise
                    
                    # Calculate batch metrics
                    batch_metrics = {
                        'loss': loss.item(),
                        'sharpe_ratio': mean_sharpe.item(),
                        'mean_return': mean_portfolio_returns.mean().item(),
                        'std_return': portfolio_stds.mean().item()
                    }
                    
                    # Track batch metrics
                    self.val_batch_tracker.add_batch(
                        batch_metrics=batch_metrics,
                        returns=portfolio_returns_timestep.cpu().numpy(),
                        weights=portfolio_weights.cpu().numpy(),
                        winner_scores=winner_scores.cpu().numpy(),
                        sharpe_ratios=sharpe_ratios.cpu().numpy()
                    )
                    
                    # Visualize episodes from first batch
                    if batch_idx == 0:
                        for ep_idx in range(min(5, portfolio_returns_timestep.shape[0])):  # Visualize up to 5 episodes
                            self.visualize_episode(
                                episode_idx=ep_idx,
                                epoch=epoch,
                                batch_idx=batch_idx,
                                phase='val',
                                returns=portfolio_returns_timestep[ep_idx].cpu().numpy(),
                                weights=portfolio_weights[ep_idx].cpu().numpy(),
                                winner_scores=winner_scores[ep_idx].cpu().numpy()
                            )
            
            # Get epoch metrics
            epoch_metrics = self.val_batch_tracker.get_epoch_metrics()
            
            # Visualize epoch
            all_returns = self.val_batch_tracker.get_all_returns()
            all_weights = self.val_batch_tracker.get_all_weights()
            all_sharpe_ratios = self.val_batch_tracker.get_all_sharpe_ratios()
            
            self.visualize_epoch(
                epoch=epoch,
                phase='val',
                all_returns=all_returns,
                all_weights=all_weights,
                all_sharpe_ratios=all_sharpe_ratios,
                metrics=epoch_metrics
            )
            
            logging.info(f"Validation Epoch {epoch}: Loss={epoch_metrics.get('loss', float('inf')):.4f}, Sharpe={epoch_metrics.get('sharpe_ratio', -float('inf')):.4f}")
            
            return epoch_metrics
            
        except Exception as e:
            logging.error(f"Error in validate: {str(e)}")
            logging.error(f"Error traceback: {traceback.format_exc()}")
            # Return empty metrics in case of error
            return {'loss': float('inf'), 'sharpe_ratio': -float('inf'), 'mean_return': 0.0}
    
    def visualize_hyperparameters(self, hyperparams_results: List[Dict[str, Any]]):
        """
        Visualize hyperparameter search results.
        
        Args:
            hyperparams_results: List of dictionaries with hyperparameter search results
        """
        if self.vis_manager is not None:
            self.vis_manager.visualize_hyperparameters(hyperparams_results)
    
    def visualize_episode(self, episode_idx: int, epoch: int, batch_idx: int, phase: str,
                        returns: np.ndarray, weights: np.ndarray, winner_scores: np.ndarray):
        """
        Visualize a single episode.
        
        Args:
            episode_idx: Episode index
            epoch: Current epoch
            batch_idx: Batch index
            phase: Training phase ('train' or 'val')
            returns: Episode returns [T]
            weights: Portfolio weights [T, num_assets]
            winner_scores: Winner scores [T, num_assets]
        """
        if self.vis_manager is not None:
            self.vis_manager.visualize_episode(
                episode_idx=episode_idx,
                epoch=epoch,
                batch_idx=batch_idx,
                phase=phase,
                returns=returns,
                weights=weights,
                winner_scores=winner_scores
            )
    
    def visualize_batch(self, batch_idx: int, epoch: int, phase: str,
                      returns: np.ndarray, weights: np.ndarray,
                      sharpe_ratios: np.ndarray, winner_scores: np.ndarray):
        """
        Visualize a batch of episodes.
        
        Args:
            batch_idx: Batch index
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
            returns: Batch returns [batch_size, T]
            weights: Portfolio weights [batch_size, T, num_assets]
            sharpe_ratios: Sharpe ratios [batch_size]
            winner_scores: Winner scores [batch_size, T, num_assets]
        """
        if self.vis_manager is not None:
            self.vis_manager.visualize_batch(
                batch_idx=batch_idx,
                epoch=epoch,
                phase=phase,
                returns=returns,
                weights=weights,
                sharpe_ratios=sharpe_ratios,
                winner_scores=winner_scores
            )
    
    def visualize_epoch(self, epoch: int, phase: str,
                       all_returns: np.ndarray, all_weights: np.ndarray,
                       all_sharpe_ratios: np.ndarray, metrics: Dict[str, float]):
        """
        Visualize an entire epoch.
        
        Args:
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
            all_returns: All returns from this epoch [num_episodes, T]
            all_weights: All portfolio weights [num_episodes, T, num_assets]
            all_sharpe_ratios: All Sharpe ratios [num_episodes]
            metrics: Dictionary of epoch metrics
        """
        if self.vis_manager is not None:
            self.vis_manager.visualize_epoch(
                epoch=epoch,
                phase=phase,
                all_returns=all_returns,
                all_weights=all_weights,
                all_sharpe_ratios=all_sharpe_ratios,
                metrics=metrics
            )
    
    def calculate_returns(self, weights: torch.Tensor, future_returns: torch.Tensor,
                        masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate portfolio returns from weights and future asset returns.
        
        Args:
            weights: Portfolio weights [batch_size, T, num_assets]
            future_returns: Future asset returns [batch_size, T, num_assets]
            masks: Optional masks for valid assets [batch_size, T, num_assets]
            
        Returns:
            Portfolio returns [batch_size, T]
        """
        batch_size, T, num_assets = weights.shape
        
        # Handle both 2D and 3D inputs
        if len(weights.shape) == 3 and len(future_returns.shape) == 3:
            # Both are 3D: [batch_size, T, num_assets]
            
            # Process each time step
            all_returns = []
            for t in range(T):
                # Extract data for this time step
                t_weights = weights[:, t]  # [batch_size, num_assets]
                t_future_returns = future_returns[:, t]  # [batch_size, num_assets]
                
                if masks is not None and len(masks.shape) == 3:
                    t_masks = masks[:, t]  # [batch_size, num_assets]
                else:
                    t_masks = masks  # [batch_size, num_assets] or None
                
                # Apply masks if provided
                if t_masks is not None:
                    t_weights = t_weights * t_masks
                
                # Calculate portfolio returns for this time step
                t_portfolio_returns = torch.sum(t_weights * t_future_returns, dim=-1)  # [batch_size]
                all_returns.append(t_portfolio_returns)
            
            # Stack returns for all time steps
            return torch.stack(all_returns, dim=1)  # [batch_size, T]
            
        else:
            # Standard case (2D inputs)
            # Apply masks if provided
            if masks is not None:
                weights = weights * masks
            
            # Calculate portfolio returns
            portfolio_returns = torch.sum(weights * future_returns, dim=-1)
            
            return portfolio_returns