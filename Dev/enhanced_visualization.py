import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import logging
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import json
from datetime import datetime

class VisualizationManager:
    """
    Comprehensive visualization manager for AlphaPortfolio.
    Creates and manages all visualizations at episode, batch, and epoch levels.
    """
    
    def _visualize_detailed_portfolio(self, episode_idx, epoch, batch_idx, weights, returns, 
                                    winner_scores, output_dir, cycle_idx):
        """Create detailed portfolio visualizations."""
        # Create output directory
        detailed_dir = os.path.join(output_dir, f"cycle_{cycle_idx}", f"epoch_{epoch}", f"episode_{episode_idx}")
        os.makedirs(detailed_dir, exist_ok=True)
        
        # 1. Plot portfolio weights at each time step
        for t in range(len(weights)):
            self._plot_weights_at_timestep(
                weights=weights[t],
                output_dir=detailed_dir,
                filename=f"weights_t{t+1}.png",
                title=f"Portfolio Weights at Time Step {t+1}"
            )
        
        # 2. Plot returns by time step
        self._plot_returns_by_timestep(
            returns=returns,
            output_dir=detailed_dir,
            filename=f"returns.png",
            title=f"Returns by Time Step"
        )
        
        # 3. Plot cumulative returns
        self._plot_cumulative_returns(
            returns=returns,
            output_dir=detailed_dir,
            filename=f"cumulative_returns.png",
            title=f"Cumulative Returns"
        )
        
        # 4. Plot portfolio allocation over time
        self._plot_portfolio_allocation_over_time(
            weights=weights,
            output_dir=detailed_dir,
            filename=f"portfolio_allocation.png",
            title=f"Portfolio Allocation Over Time"
        )
    
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
    
    def __init__(self, output_dir: str, cycle_idx: int, T: int, num_assets: int, assets_list: Optional[List[str]] = None):
        """
        Initialize visualization manager.
        
        Args:
            output_dir: Output directory for plots
            cycle_idx: Current training cycle index
            T: Number of time steps per episode
            num_assets: Number of assets in the universe
            assets_list: Optional list of asset names/identifiers
        """
        self.output_dir = output_dir
        self.cycle_idx = cycle_idx
        self.T = T
        self.num_assets = num_assets
        self.assets_list = assets_list if assets_list else [f"Asset {i+1}" for i in range(num_assets)]
        
        # Create output directory for this cycle
        self.cycle_dir = os.path.join(output_dir, f"cycle_{cycle_idx}")
        os.makedirs(self.cycle_dir, exist_ok=True)
        
        # Create subdirectories for different visualization levels
        self.episode_dir = os.path.join(self.cycle_dir, "episodes")
        self.batch_dir = os.path.join(self.cycle_dir, "batches")
        self.epoch_dir = os.path.join(self.cycle_dir, "epochs")
        
        for dir_path in [self.episode_dir, self.batch_dir, self.epoch_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        logging.info(f"Initialized visualization manager for cycle {cycle_idx}")
        logging.info(f"Output directory: {self.cycle_dir}")
        
    def visualize_episode(self, episode_idx: int, epoch: int, batch_idx: int, phase: str,
                        returns: np.ndarray, weights: np.ndarray, 
                        winner_scores: Optional[np.ndarray] = None):
        """
        Create visualizations for a single episode.
        
        Args:
            episode_idx: Episode index
            epoch: Current epoch
            batch_idx: Batch index
            phase: Training phase ('train' or 'val')
            returns: Episode returns [T]
            weights: Portfolio weights [T, num_assets]
            winner_scores: Optional winner scores [T, num_assets]
        """
        # Create episode subdirectory
        episode_subdir = os.path.join(self.episode_dir, f"epoch_{epoch}_{phase}")
        os.makedirs(episode_subdir, exist_ok=True)
        
        # Base filename
        base_filename = f"ep_{episode_idx}_batch_{batch_idx}"
        
        # Plot episode returns
        self._plot_episode_returns(
            returns=returns,
            output_dir=episode_subdir,
            filename=f"{base_filename}_returns.png",
            title=f"Episode {episode_idx} Returns (Epoch {epoch}, Batch {batch_idx})"
        )
        
        # Plot episode portfolio allocation
        self._plot_episode_portfolio_allocation(
            weights=weights,
            output_dir=episode_subdir,
            filename=f"{base_filename}_allocation.png",
            title=f"Episode {episode_idx} Portfolio Allocation (Epoch {epoch}, Batch {batch_idx})"
        )
        
        # Plot episode cumulative returns
        self._plot_episode_cumulative_returns(
            returns=returns,
            output_dir=episode_subdir,
            filename=f"{base_filename}_cumreturns.png",
            title=f"Episode {episode_idx} Cumulative Returns (Epoch {epoch}, Batch {batch_idx})"
        )
        
        # Plot episode asset selection
        self._plot_episode_asset_selection(
            weights=weights,
            winner_scores=winner_scores,
            output_dir=episode_subdir,
            filename=f"{base_filename}_selection.png",
            title=f"Episode {episode_idx} Asset Selection (Epoch {epoch}, Batch {batch_idx})"
        )
        
    def visualize_batch(self, batch_idx: int, epoch: int, phase: str,
                      returns: np.ndarray, weights: np.ndarray,
                      sharpe_ratios: Optional[np.ndarray] = None,
                      winner_scores: Optional[np.ndarray] = None):
        """
        Create visualizations for a batch of episodes.
        
        Args:
            batch_idx: Batch index
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
            returns: Batch returns [batch_size, T]
            weights: Portfolio weights [batch_size, T, num_assets]
            sharpe_ratios: Optional Sharpe ratios [batch_size]
            winner_scores: Optional winner scores [batch_size, T, num_assets]
        """
        # Create batch subdirectory
        batch_subdir = os.path.join(self.batch_dir, f"epoch_{epoch}_{phase}")
        os.makedirs(batch_subdir, exist_ok=True)
        
        # Base filename
        base_filename = f"batch_{batch_idx}"
        
        # Plot batch returns
        self._plot_batch_returns(
            returns=returns,
            output_dir=batch_subdir,
            filename=f"{base_filename}_returns.png",
            title=f"Batch {batch_idx} Returns (Epoch {epoch})"
        )
        
        # Plot batch Sharpe ratios if available
        if sharpe_ratios is not None:
            self._plot_batch_sharpe_ratios(
                sharpe_ratios=sharpe_ratios,
                output_dir=batch_subdir,
                filename=f"{base_filename}_sharpe.png",
                title=f"Batch {batch_idx} Sharpe Ratios (Epoch {epoch})"
            )
        
        # Plot batch aggregated portfolio allocation
        self._plot_batch_portfolio_allocation(
            weights=weights,
            output_dir=batch_subdir,
            filename=f"{base_filename}_allocation.png",
            title=f"Batch {batch_idx} Portfolio Allocation (Epoch {epoch})"
        )
        
        # Plot batch heatmap of returns
        self._plot_batch_returns_heatmap(
            returns=returns,
            output_dir=batch_subdir,
            filename=f"{base_filename}_returns_heatmap.png",
            title=f"Batch {batch_idx} Returns Heatmap (Epoch {epoch})"
        )
        
        # Generate batch summary statistics
        self._save_batch_statistics(
            returns=returns,
            sharpe_ratios=sharpe_ratios,
            output_dir=batch_subdir,
            filename=f"{base_filename}_stats.json"
        )
    
    def visualize_epoch(self, epoch: int, phase: str,
                       all_returns: np.ndarray, all_weights: np.ndarray,
                       all_sharpe_ratios: Optional[np.ndarray] = None,
                       metrics: Optional[Dict[str, float]] = None):
        """
        Create visualizations for an entire epoch.
        
        Args:
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
            all_returns: All returns from this epoch [num_episodes, T]
            all_weights: All portfolio weights [num_episodes, T, num_assets]
            all_sharpe_ratios: Optional all Sharpe ratios [num_episodes]
            metrics: Optional dictionary of epoch metrics
        """
        # Base filename
        base_filename = f"epoch_{epoch}_{phase}"
        
        # Plot epoch returns distribution
        self._plot_epoch_returns_distribution(
            returns=all_returns,
            output_dir=self.epoch_dir,
            filename=f"{base_filename}_returns_dist.png",
            title=f"Epoch {epoch} Returns Distribution ({phase.capitalize()})"
        )
        
        # Plot epoch returns by time step
        self._plot_epoch_returns_by_timestep(
            returns=all_returns,
            output_dir=self.epoch_dir,
            filename=f"{base_filename}_returns_timestep.png",
            title=f"Epoch {epoch} Returns by Time Step ({phase.capitalize()})"
        )
        
        # Plot epoch Sharpe ratios if available
        if all_sharpe_ratios is not None:
            self._plot_epoch_sharpe_distribution(
                sharpe_ratios=all_sharpe_ratios,
                output_dir=self.epoch_dir,
                filename=f"{base_filename}_sharpe_dist.png",
                title=f"Epoch {epoch} Sharpe Ratio Distribution ({phase.capitalize()})"
            )
        
        # Plot epoch portfolio allocations
        self._plot_epoch_portfolio_allocation(
            weights=all_weights,
            output_dir=self.epoch_dir,
            filename=f"{base_filename}_allocation.png",
            title=f"Epoch {epoch} Portfolio Allocation ({phase.capitalize()})"
        )
        
        # Plot epoch cumulative returns
        self._plot_epoch_cumulative_returns(
            returns=all_returns,
            output_dir=self.epoch_dir,
            filename=f"{base_filename}_cumreturns.png",
            title=f"Epoch {epoch} Cumulative Returns ({phase.capitalize()})"
        )
        
        # Save epoch metrics if available
        if metrics:
            self._save_epoch_metrics(
                metrics=metrics,
                output_dir=self.epoch_dir,
                filename=f"{base_filename}_metrics.json"
            )
    
    def visualize_hyperparameters(self, hyperparams_results: List[Dict[str, Any]],
                                output_filename: str = "hyperparameter_analysis.png"):
        """
        Create visualizations for hyperparameter search results.
        
        Args:
            hyperparams_results: List of dictionaries with hyperparameter search results
            output_filename: Output filename
        """
        if not hyperparams_results:
            logging.warning("No hyperparameter results to visualize")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(hyperparams_results)
        
        # Create figure with multiple subplots
        num_params = len(df.columns) - 1  # Exclude val_sharpe
        rows = (num_params + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        axes = axes.flatten()
        
        # Plot each hyperparameter vs. validation Sharpe
        param_idx = 0
        for col in df.columns:
            if col != 'val_sharpe':
                ax = axes[param_idx]
                
                # Group by this hyperparameter
                grouped = df.groupby(col)['val_sharpe'].mean().reset_index()
                
                # Plot
                ax.bar(grouped[col].astype(str), grouped['val_sharpe'], color='skyblue')
                ax.set_title(f"{col} vs. Validation Sharpe", fontsize=14)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Mean Validation Sharpe', fontsize=12)
                ax.grid(axis='y')
                
                param_idx += 1
        
        # Hide any unused subplots
        for i in range(param_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.cycle_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved hyperparameter analysis to {output_path}")
        
        # Create interaction plots for pairs of hyperparameters
        self._plot_hyperparameter_interactions(df, output_filename="hyperparameter_interactions.png")
    
    def visualize_training_progress(self, metrics: Dict[str, List[float]],
                                  output_filename: str = "training_progress.png"):
        """
        Create visualizations for training progress across epochs.
        
        Args:
            metrics: Dictionary of metrics lists
            output_filename: Output filename
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot Sharpe ratio
        ax1 = axes[0, 0]
        if 'train_sharpe' in metrics and metrics['train_sharpe']:
            ax1.plot(metrics['train_sharpe'], 'b-', label='Train')
        if 'val_sharpe' in metrics and metrics['val_sharpe']:
            ax1.plot(metrics['val_sharpe'], 'r-', label='Validation')
        ax1.set_title('Sharpe Ratio', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.legend()
        ax1.grid(True)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Plot loss
        ax2 = axes[0, 1]
        if 'train_loss' in metrics and metrics['train_loss']:
            ax2.plot(metrics['train_loss'], 'b-', label='Train')
        if 'val_loss' in metrics and metrics['val_loss']:
            ax2.plot(metrics['val_loss'], 'r-', label='Validation')
        ax2.set_title('Loss', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Plot returns
        ax3 = axes[1, 0]
        if 'train_returns' in metrics and metrics['train_returns']:
            ax3.plot(metrics['train_returns'], 'b-', label='Train')
        if 'val_returns' in metrics and metrics['val_returns']:
            ax3.plot(metrics['val_returns'], 'r-', label='Validation')
        ax3.set_title('Mean Return', fontsize=14)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Return', fontsize=12)
        ax3.legend()
        ax3.grid(True)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Plot portfolio weights
        ax4 = axes[1, 1]
        if 'mean_weight_long' in metrics and metrics['mean_weight_long']:
            ax4.plot(metrics['mean_weight_long'], 'g-', label='Long')
        if 'mean_weight_short' in metrics and metrics['mean_weight_short']:
            ax4.plot(metrics['mean_weight_short'], 'r-', label='Short')
        ax4.set_title('Mean Portfolio Weights', fontsize=14)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Weight', fontsize=12)
        ax4.legend()
        ax4.grid(True)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.cycle_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved training progress plot to {output_path}")
    
    def _plot_episode_returns(self, returns: np.ndarray, output_dir: str, 
                            filename: str, title: str):
        """Plot returns for a single episode."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(returns) + 1), returns * 100, 'b-o')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(True)
        plt.xticks(range(1, len(returns) + 1))
        
        # Add mean return line
        mean_return = np.mean(returns) * 100
        plt.axhline(y=mean_return, color='g', linestyle='-', alpha=0.5, 
                   label=f'Mean: {mean_return:.2f}%')
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_episode_portfolio_allocation(self, weights: np.ndarray, output_dir: str, 
                                         filename: str, title: str):
        """Plot portfolio allocation for a single episode."""
        # Make sure we can plot something even if there are very few weights
        if weights.shape[1] == 0:
            logging.warning("No portfolio weights to plot")
            return
        
        # Calculate mean absolute weights to find important assets
        mean_abs_weights = np.mean(np.abs(weights), axis=0)
        
        # Make sure assets_list exists and has enough elements
        if not hasattr(self, 'assets_list') or self.assets_list is None:
            # Create generic asset names
            self.assets_list = [f"Asset {i+1}" for i in range(len(mean_abs_weights))]
        elif len(self.assets_list) < len(mean_abs_weights):
            # Add missing asset names
            missing_count = len(mean_abs_weights) - len(self.assets_list)
            logging.warning(f"Adding {missing_count} missing asset names. This may indicate a mismatch between data and asset list.")
            start_idx = len(self.assets_list)
            self.assets_list.extend([f"Asset {i+1}" for i in range(start_idx, start_idx + missing_count)])
        
        # Get indices of top assets (up to 15, but limited by actual asset count)
        max_assets_to_display = min(15, len(mean_abs_weights))
        top_indices = np.argsort(-mean_abs_weights)[:max_assets_to_display]
        
        # Double-check that indices are within bounds
        top_indices = [idx for idx in top_indices if idx < len(self.assets_list)]
        
        # If no valid indices found, return early
        if not top_indices:
            logging.warning("No valid top indices found for portfolio allocation plot")
            return
        
        # Select weights and asset names
        selected_weights = weights[:, top_indices]
        selected_assets = [self.assets_list[i] for i in top_indices]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        try:
            # Create heatmap
            sns.heatmap(
                selected_weights.T,  # Transpose to have assets as rows
                cmap='RdBu_r',
                center=0,
                vmin=-np.max(np.abs(selected_weights) + 1e-10),  # Add small constant to avoid division by zero
                vmax=np.max(np.abs(selected_weights) + 1e-10),
                yticklabels=selected_assets,
                xticklabels=[f"t={i+1}" for i in range(len(weights))],
                cbar_kws={'label': 'Portfolio Weight'}
            )
            
            plt.title(title, fontsize=14)
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Asset', fontsize=12)
            
            # Save figure
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            logging.error(f"Error creating portfolio allocation plot: {str(e)}")
        finally:
            plt.close()
   
    def _plot_episode_cumulative_returns(self, returns: np.ndarray, output_dir: str, 
                                       filename: str, title: str):
        """Plot cumulative returns for a single episode."""
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(returns) + 1), cum_returns * 100, 'b-o')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True)
        plt.xticks(range(1, len(returns) + 1))
        
        # Add final return annotation
        final_return = cum_returns[-1] * 100
        plt.annotate(f"Final: {final_return:.2f}%", 
                    xy=(len(returns), final_return), 
                    xytext=(len(returns) - 1, final_return + 1),
                    arrowprops=dict(arrowstyle='->'))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_episode_asset_selection(self, weights: np.ndarray, 
                                    winner_scores: Optional[np.ndarray], 
                                    output_dir: str, filename: str, title: str):
        """Plot asset selection over time for a single episode."""
        plt.figure(figsize=(12, 10))
        
        # Create subplots
        gs = GridSpec(2, 2, figure=plt.gcf())
        ax1 = plt.subplot(gs[0, 0])  # Top long positions
        ax2 = plt.subplot(gs[0, 1])  # Top short positions
        ax3 = plt.subplot(gs[1, :])  # Position counts over time
        
        # Calculate long and short positions for each time step
        long_counts = np.sum(weights > 0, axis=1)
        short_counts = np.sum(weights < 0, axis=1)
        
        # Find the time step with the most interesting allocation for detailed view
        interesting_step = np.argmax(np.abs(weights).sum(axis=1))
        
        # Plot top long positions for the interesting time step
        step_weights = weights[interesting_step]
        long_indices = np.where(step_weights > 0)[0]
        if len(long_indices) > 0:
            long_values = step_weights[long_indices]
            long_assets = [self.assets_list[i] for i in long_indices]
            
            # Sort by weight magnitude
            long_order = np.argsort(-long_values)
            top_long_assets = [long_assets[i] for i in long_order[:10]]
            top_long_values = long_values[long_order[:10]]
            
            ax1.barh(top_long_assets, top_long_values, color='green')
            ax1.set_title(f'Top Long Positions (t={interesting_step+1})', fontsize=12)
            ax1.set_xlabel('Weight', fontsize=10)
            ax1.grid(axis='x')
        
        # Plot top short positions for the interesting time step
        short_indices = np.where(step_weights < 0)[0]
        if len(short_indices) > 0:
            short_values = step_weights[short_indices]
            short_assets = [self.assets_list[i] for i in short_indices]
            
            # Sort by weight magnitude
            short_order = np.argsort(short_values)
            top_short_assets = [short_assets[i] for i in short_order[:10]]
            top_short_values = short_values[short_order[:10]]
            
            ax2.barh(top_short_assets, top_short_values, color='red')
            ax2.set_title(f'Top Short Positions (t={interesting_step+1})', fontsize=12)
            ax2.set_xlabel('Weight', fontsize=10)
            ax2.grid(axis='x')
        
        # Plot position counts over time
        x = range(1, len(weights) + 1)
        ax3.bar(x, long_counts, color='green', alpha=0.7, label='Long Positions')
        ax3.bar(x, -short_counts, color='red', alpha=0.7, label='Short Positions')
        ax3.set_title('Number of Positions Over Time', fontsize=12)
        ax3.set_xlabel('Time Step', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.legend()
        ax3.grid(axis='y')
        ax3.set_xticks(x)
        
        # Add zero line for reference
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_returns(self, returns: np.ndarray, output_dir: str, 
                          filename: str, title: str):
        """Plot returns for a batch of episodes."""
        # Calculate mean and std returns for each time step
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        
        plt.figure(figsize=(12, 8))
        
        # Plot mean returns
        plt.plot(range(1, self.T + 1), mean_returns * 100, 'b-', linewidth=2, label='Mean Return')
        
        # Add error bands
        plt.fill_between(
            range(1, self.T + 1),
            (mean_returns - std_returns) * 100,
            (mean_returns + std_returns) * 100,
            color='blue',
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.xticks(range(1, self.T + 1))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_sharpe_ratios(self, sharpe_ratios: np.ndarray, output_dir: str, 
                                filename: str, title: str):
        """Plot Sharpe ratios for a batch of episodes."""
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of Sharpe ratios
        sns.histplot(sharpe_ratios, kde=True, bins=20, color='skyblue')
        
        # Add mean and median lines
        mean_sharpe = np.mean(sharpe_ratios)
        median_sharpe = np.median(sharpe_ratios)
        plt.axvline(x=mean_sharpe, color='red', linestyle='-', 
                   label=f'Mean: {mean_sharpe:.4f}')
        plt.axvline(x=median_sharpe, color='green', linestyle='--', 
                   label=f'Median: {median_sharpe:.4f}')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Sharpe Ratio', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_portfolio_allocation(self, weights: np.ndarray, output_dir: str, 
                                       filename: str, title: str):
        """Plot aggregated portfolio allocation for a batch of episodes."""
        # Calculate mean absolute weights across all episodes in this batch
        mean_abs_weights = np.mean(np.mean(np.abs(weights), axis=0), axis=0)
        
        # Find top assets by weight magnitude
        top_indices = np.argsort(-mean_abs_weights)[:20]
        
        # Calculate average weights for top assets across episodes and time steps
        mean_weights = np.mean(np.mean(weights, axis=0), axis=0)[top_indices]
        
        # Get asset names
        top_assets = [self.assets_list[i] for i in top_indices]
        
        # Create sorted version for visualization
        sorted_indices = np.argsort(mean_weights)
        sorted_assets = [top_assets[i] for i in sorted_indices]
        sorted_weights = mean_weights[sorted_indices]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create colors based on sign
        colors = ['red' if w < 0 else 'green' for w in sorted_weights]
        
        # Plot horizontal bar chart
        plt.barh(sorted_assets, sorted_weights, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Mean Portfolio Weight', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_returns_heatmap(self, returns: np.ndarray, output_dir: str, 
                                  filename: str, title: str):
        """Plot heatmap of returns for a batch of episodes."""
        plt.figure(figsize=(14, 10))
        
        # Limit to first 30 episodes for readability
        max_episodes = min(30, returns.shape[0])
        display_returns = returns[:max_episodes]
        
        # Create heatmap
        sns.heatmap(
            display_returns * 100,  # Convert to percentages
            cmap='RdBu_r',
            center=0,
            vmin=-max(1, np.max(np.abs(display_returns)) * 100),  # At least ±1%
            vmax=max(1, np.max(np.abs(display_returns)) * 100),
            yticklabels=[f"Episode {i+1}" for i in range(display_returns.shape[0])],
            xticklabels=[f"t={i+1}" for i in range(self.T)],
            cbar_kws={'label': 'Return (%)'}
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Episode', fontsize=12)
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_batch_statistics(self, returns: np.ndarray, sharpe_ratios: Optional[np.ndarray],
                             output_dir: str, filename: str):
        """Save batch statistics to JSON file."""
        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_return_by_step = np.mean(returns, axis=0).tolist()
        std_return_by_step = np.std(returns, axis=0).tolist()
        
        # Calculate correlation matrix between time steps
        corr_matrix = np.corrcoef(returns.T).tolist()
        
        # Prepare statistics dictionary
        stats = {
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'mean_return_by_step': mean_return_by_step,
            'std_return_by_step': std_return_by_step,
            'return_correlation_matrix': corr_matrix,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add Sharpe ratio statistics if available
        if sharpe_ratios is not None:
            stats.update({
                'mean_sharpe': float(np.mean(sharpe_ratios)),
                'median_sharpe': float(np.median(sharpe_ratios)),
                'min_sharpe': float(np.min(sharpe_ratios)),
                'max_sharpe': float(np.max(sharpe_ratios)),
                'std_sharpe': float(np.std(sharpe_ratios))
            })
        
        # Save to JSON file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)
    
    def _plot_epoch_returns_distribution(self, returns: np.ndarray, output_dir: str, 
                                       filename: str, title: str):
        """Plot distribution of returns for an epoch."""
        # Flatten returns across all episodes and time steps
        flat_returns = returns.flatten()
        
        plt.figure(figsize=(12, 8))
        
        # Create main distribution plot
        sns.histplot(flat_returns * 100, kde=True, bins=50, color='skyblue')
        
        # Add vertical lines for key statistics
        mean_return = np.mean(flat_returns) * 100
        median_return = np.median(flat_returns) * 100
        plt.axvline(x=mean_return, color='red', linestyle='-', 
                   label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=median_return, color='green', linestyle='--', 
                   label=f'Median: {median_return:.2f}%')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add key statistics as text
        stats_text = (
            f"Mean: {mean_return:.2f}%\n"
            f"Median: {median_return:.2f}%\n"
            f"Std Dev: {np.std(flat_returns) * 100:.2f}%\n"
            f"Min: {np.min(flat_returns) * 100:.2f}%\n"
            f"Max: {np.max(flat_returns) * 100:.2f}%\n"
            f"Positive Returns: {np.mean(flat_returns > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_epoch_returns_by_timestep(self, returns: np.ndarray, output_dir: str, 
                                      filename: str, title: str):
        """Plot returns by time step for an epoch."""
        # Calculate statistics for each time step
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        median_returns = np.median(returns, axis=0)
        q25_returns = np.percentile(returns, 25, axis=0)
        q75_returns = np.percentile(returns, 75, axis=0)
        
        plt.figure(figsize=(14, 10))
        
        # Create a boxplot for each time step
        ax = plt.subplot(2, 1, 1)
        ax.boxplot(returns * 100, notch=True, patch_artist=True,
                 boxprops=dict(facecolor='skyblue', alpha=0.7),
                 medianprops=dict(color='navy'),
                 flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
        
        ax.set_title('Return Distribution by Time Step', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticklabels([f"t={i+1}" for i in range(self.T)])
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot mean and confidence intervals
        ax2 = plt.subplot(2, 1, 2)
        x = np.arange(1, self.T + 1)
        
        # Plot mean line
        ax2.plot(x, mean_returns * 100, 'b-', linewidth=2, label='Mean')
        
        # Add error bands
        ax2.fill_between(
            x,
            (mean_returns - std_returns) * 100,
            (mean_returns + std_returns) * 100,
            color='blue',
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        # Add IQR bands
        ax2.fill_between(
            x,
            q25_returns * 100,
            q75_returns * 100,
            color='green',
            alpha=0.2,
            label='IQR (25-75%)'
        )
        
        # Add median line
        ax2.plot(x, median_returns * 100, 'g--', linewidth=1.5, label='Median')
        
        # Add zero line
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_title('Mean Return by Time Step with Variability', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        ax2.set_xticks(x)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_epoch_sharpe_distribution(self, sharpe_ratios: np.ndarray, output_dir: str, 
                                      filename: str, title: str):
        """Plot distribution of Sharpe ratios for an epoch."""
        plt.figure(figsize=(12, 8))
        
        # Create main distribution plot
        sns.histplot(sharpe_ratios, kde=True, bins=30, color='skyblue')
        
        # Add vertical lines for key statistics
        mean_sharpe = np.mean(sharpe_ratios)
        median_sharpe = np.median(sharpe_ratios)
        plt.axvline(x=mean_sharpe, color='red', linestyle='-', 
                   label=f'Mean: {mean_sharpe:.4f}')
        plt.axvline(x=median_sharpe, color='green', linestyle='--', 
                   label=f'Median: {median_sharpe:.4f}')
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel('Sharpe Ratio', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add key statistics as text
        stats_text = (
            f"Mean: {mean_sharpe:.4f}\n"
            f"Median: {median_sharpe:.4f}\n"
            f"Std Dev: {np.std(sharpe_ratios):.4f}\n"
            f"Min: {np.min(sharpe_ratios):.4f}\n"
            f"Max: {np.max(sharpe_ratios):.4f}\n"
            f"Positive Sharpe: {np.mean(sharpe_ratios > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_epoch_portfolio_allocation(self, weights: np.ndarray, output_dir: str, 
                                       filename: str, title: str):
        """Plot portfolio allocation for an epoch."""
        # Calculate mean weights across all episodes and time steps
        mean_weights = np.mean(np.mean(weights, axis=0), axis=0)
        
        # Calculate mean absolute weights to find important assets
        mean_abs_weights = np.mean(np.mean(np.abs(weights), axis=0), axis=0)
        
        # Find top assets by weight magnitude
        top_indices = np.argsort(-mean_abs_weights)[:20]
        top_assets = [self.assets_list[i] for i in top_indices]
        top_weights = mean_weights[top_indices]
        
        # Create figure with multiple subplots
        plt.figure(figsize=(14, 12))
        gs = GridSpec(2, 2, figure=plt.gcf())
        
        # Plot 1: Top assets by mean weight
        ax1 = plt.subplot(gs[0, :])
        
        # Sort by weight
        sorted_indices = np.argsort(top_weights)
        sorted_assets = [top_assets[i] for i in sorted_indices]
        sorted_weights = top_weights[sorted_indices]
        
        # Set colors based on sign
        colors = ['red' if w < 0 else 'green' for w in sorted_weights]
        
        ax1.barh(sorted_assets, sorted_weights, color=colors)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('Top Assets by Mean Weight', fontsize=14)
        ax1.set_xlabel('Mean Weight', fontsize=12)
        ax1.set_ylabel('Asset', fontsize=12)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Number of assets used over time
        ax2 = plt.subplot(gs[1, 0])
        
        # Calculate mean number of long and short positions for each time step
        long_counts = np.mean(np.sum(weights > 0, axis=2), axis=0)
        short_counts = np.mean(np.sum(weights < 0, axis=2), axis=0)
        
        x = range(1, self.T + 1)
        ax2.bar(x, long_counts, color='green', alpha=0.7, label='Long Positions')
        ax2.bar(x, -short_counts, color='red', alpha=0.7, label='Short Positions')
        ax2.set_title('Mean Number of Positions by Time Step', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(x)
        
        # Add zero line for reference
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot 3: Weight stability (standard deviation of weights for top assets)
        ax3 = plt.subplot(gs[1, 1])
        
        # Calculate std dev of weights across episodes and time steps
        std_weights = np.std(np.mean(weights, axis=1), axis=0)[top_indices]
        
        # Create a scatter plot: mean weight vs std dev
        ax3.scatter(top_weights, std_weights, c=colors, s=100, alpha=0.7)
        
        # Add asset labels
        for i, asset in enumerate(top_assets):
            ax3.annotate(asset, (top_weights[i], std_weights[i]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8)
        
        ax3.set_title('Weight Stability Analysis', fontsize=14)
        ax3.set_xlabel('Mean Weight', fontsize=12)
        ax3.set_ylabel('Weight Standard Deviation', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_epoch_cumulative_returns(self, returns: np.ndarray, output_dir: str, 
                                     filename: str, title: str):
        """Plot cumulative returns paths for an epoch."""
        # Calculate cumulative returns for each episode
        cum_returns = np.cumprod(1 + returns, axis=1) - 1
        
        # Calculate statistics
        mean_cum_returns = np.mean(cum_returns, axis=0)
        std_cum_returns = np.std(cum_returns, axis=0)
        median_cum_returns = np.median(cum_returns, axis=0)
        q25_cum_returns = np.percentile(cum_returns, 25, axis=0)
        q75_cum_returns = np.percentile(cum_returns, 75, axis=0)
        
        plt.figure(figsize=(14, 10))
        
        # Plot individual paths for a sample of episodes
        max_paths = min(50, cum_returns.shape[0])
        sample_indices = np.random.choice(cum_returns.shape[0], max_paths, replace=False)
        
        for i in sample_indices:
            plt.plot(range(1, self.T + 1), cum_returns[i] * 100, 'b-', alpha=0.1)
        
        # Plot mean path
        plt.plot(range(1, self.T + 1), mean_cum_returns * 100, 'r-', linewidth=2, label='Mean')
        
        # Plot median path
        plt.plot(range(1, self.T + 1), median_cum_returns * 100, 'g-', linewidth=2, label='Median')
        
        # Add IQR bands
        plt.fill_between(
            range(1, self.T + 1),
            q25_cum_returns * 100,
            q75_cum_returns * 100,
            color='grey',
            alpha=0.3,
            label='IQR (25-75%)'
        )
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add annotations
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, self.T + 1))
        
        # Add final return statistics
        final_mean = mean_cum_returns[-1] * 100
        final_median = median_cum_returns[-1] * 100
        stats_text = (
            f"Final Mean: {final_mean:.2f}%\n"
            f"Final Median: {final_median:.2f}%\n"
            f"Positive Final: {np.mean(cum_returns[:, -1] > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_epoch_metrics(self, metrics: Dict[str, float], output_dir: str, filename: str):
        """Save epoch metrics to JSON file."""
        # Add timestamp
        metrics_with_timestamp = {
            **metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to JSON file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=4)
    
    def _plot_hyperparameter_interactions(self, df: pd.DataFrame, 
                                        output_filename: str = "hyperparameter_interactions.png"):
        """Plot interactions between hyperparameters."""
        # Identify hyperparameter columns (exclude val_sharpe)
        param_cols = [col for col in df.columns if col != 'val_sharpe']
        
        # Create combinations of hyperparameters
        import itertools
        combinations = list(itertools.combinations(param_cols, 2))
        
        # Limit to at most 6 combinations for readability
        if len(combinations) > 6:
            combinations = combinations[:6]
        
        # Create figure with subplots
        rows = (len(combinations) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows))
        axes = axes.flatten()
        
        for i, (param1, param2) in enumerate(combinations):
            ax = axes[i]
            
            # Create pivot table
            pivot = df.pivot_table(
                values='val_sharpe',
                index=param1,
                columns=param2,
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot,
                cmap='viridis',
                annot=True,
                fmt=".4f",
                linewidths=.5,
                ax=ax,
                cbar_kws={'label': 'Mean Validation Sharpe'}
            )
            
            ax.set_title(f"{param1} vs {param2} (Effect on Sharpe)", fontsize=14)
        
        # Hide any unused subplots
        for i in range(len(combinations), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Hyperparameter Interactions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save figure
        output_path = os.path.join(self.cycle_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved hyperparameter interactions to {output_path}")
        
    def plot_portfolio_std_over_time(self, std_values: np.ndarray, output_dir: str, 
                                    filename: str = "portfolio_std_over_time.png", 
                                    title: str = "Portfolio Standard Deviation Over Time"):
        """
        Plot portfolio standard deviation over time.
        
        Args:
            std_values: Standard deviation values [num_episodes, T] or [T]
            output_dir: Output directory
            filename: Output filename
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Check if we have multiple episodes or just one
        if std_values.ndim == 2:
            # Multiple episodes
            num_episodes, T = std_values.shape
            
            # Plot mean and std bands
            mean_std = np.mean(std_values, axis=0)
            std_of_std = np.std(std_values, axis=0)
            
            x = np.arange(1, T + 1)
            plt.plot(x, mean_std * 100, 'b-', linewidth=2, label='Mean')
            
            plt.fill_between(
                x,
                (mean_std - std_of_std) * 100,
                (mean_std + std_of_std) * 100,
                color='blue',
                alpha=0.2,
                label='±1 Std Dev'
            )
            
            # Plot individual episode std (up to 20 for clarity)
            for i in range(min(20, num_episodes)):
                plt.plot(x, std_values[i] * 100, 'gray', alpha=0.3)
        else:
            # Single episode
            T = len(std_values)
            x = np.arange(1, T + 1)
            plt.plot(x, std_values * 100, 'b-o', linewidth=2)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Portfolio Standard Deviation (%)', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sharpe_evolution(self, sharpe_values: np.ndarray, output_dir: str, 
                            filename: str = "sharpe_evolution.png", 
                            title: str = "Sharpe Ratio Evolution"):
        """
        Plot evolution of Sharpe ratios over time or across episodes.
        
        Args:
            sharpe_values: Sharpe ratio values [num_episodes] or [num_epochs]
            output_dir: Output directory
            filename: Output filename
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        x = np.arange(1, len(sharpe_values) + 1)
        plt.plot(x, sharpe_values, 'g-o', linewidth=2)
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Add horizontal line at mean Sharpe
        mean_sharpe = np.mean(sharpe_values)
        plt.axhline(y=mean_sharpe, color='b', linestyle='-', alpha=0.5, 
                label=f'Mean: {mean_sharpe:.4f}')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Episode/Epoch', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sharpe_heatmap(self, sharpe_values: np.ndarray, output_dir: str, 
                        filename: str = "sharpe_heatmap.png", 
                        title: str = "Sharpe Ratio Heatmap"):
        """
        Plot heatmap of Sharpe ratios across episodes and time steps.
        
        Args:
            sharpe_values: Sharpe ratio values [num_episodes, T]
            output_dir: Output directory
            filename: Output filename
            title: Plot title
        """
        if sharpe_values.ndim != 2:
            return  # Only works for 2D data
        
        plt.figure(figsize=(14, 10))
        
        # Limit to first 30 episodes for readability
        max_episodes = min(30, sharpe_values.shape[0])
        display_values = sharpe_values[:max_episodes]
        
        # Create heatmap
        sns.heatmap(
            display_values,
            cmap='RdYlGn',
            center=0,
            vmin=-max(1, np.max(np.abs(display_values))),
            vmax=max(1, np.max(np.abs(display_values))),
            yticklabels=[f"Episode {i+1}" for i in range(display_values.shape[0])],
            xticklabels=[f"t={i+1}" for i in range(display_values.shape[1])],
            cbar_kws={'label': 'Sharpe Ratio'}
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Episode', fontsize=12)
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sharpe_vs_std(self, sharpe_values: np.ndarray, std_values: np.ndarray, 
                        output_dir: str, filename: str = "sharpe_vs_std.png", 
                        title: str = "Sharpe Ratio vs Standard Deviation"):
        """
        Plot Sharpe ratio against standard deviation for risk-return analysis.
        
        Args:
            sharpe_values: Sharpe ratio values [num_episodes]
            std_values: Standard deviation values [num_episodes]
            output_dir: Output directory
            filename: Output filename
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(std_values * 100, sharpe_values, c=sharpe_values, 
                cmap='viridis', alpha=0.7, s=80)
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Sharpe Ratio', fontsize=12)
        
        # Add mean lines
        plt.axvline(x=np.mean(std_values) * 100, color='r', linestyle='--', alpha=0.5, 
                label=f'Mean Std Dev: {np.mean(std_values)*100:.2f}%')
        plt.axhline(y=np.mean(sharpe_values), color='g', linestyle='--', alpha=0.5, 
                label=f'Mean Sharpe: {np.mean(sharpe_values):.4f}')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Portfolio Standard Deviation (%)', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
