import os
from scipy import stats
from sqlalchemy import JSON, null
import torch
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import traceback
import seaborn as sns
from enhanced_training import calculate_returns
from enhanced_visualization import *

class TestEvaluator:
    """
    Comprehensive test evaluator for AlphaPortfolio.
    Handles evaluation, benchmarking, and visualization of test results.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        config,
        device: torch.device,
        vis_manager = None
    ):
        """
        Initialize test evaluator.
        
        Args:
            model: Trained AlphaPortfolio model
            test_loader: Data loader for test data
            config: Configuration object
            device: Device to evaluate on
            vis_manager: Optional visualization manager
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.vis_manager = vis_manager
        
        # Import this here to avoid circular imports
        
        self.calculate_returns = calculate_returns
        
        # Create output directories
        self.output_dir = config.config["paths"]["output_dir"]
        self.test_dir = os.path.join(self.output_dir, "test_results")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Get test configuration
        self.test_config = config.config.get("test", {})
        self.transaction_cost_rate = self.test_config.get("transaction_cost_rate", 0.001)
        self.bootstrap_iterations = self.test_config.get("bootstrap_iterations", 1000)
        self.compare_benchmark = self.test_config.get("compare_benchmark", True)
        
        # Load asset names if available
        self.assets_list = None
        if hasattr(test_loader.dataset, 'get_asset_names'):
            self.assets_list = test_loader.dataset.get_asset_names()
        
        logging.info(f"Initialized test evaluator")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logging.info("Starting model evaluation on test data")
        
        self.model.eval()
        
        # Initialize results storage
        all_portfolio_returns_step = []  # List of [batch_size, T] tensors
        all_portfolio_weights = []       # List of [batch_size, T, num_assets] tensors
        all_winner_scores = []           # List of [batch_size, T, num_assets] tensors
        all_episode_returns = []         # List of [T] arrays for each episode
        all_episode_weights = []         # List of [T, num_assets] arrays for each episode
        
        # Track total episodes
        total_episodes = 0
        
        # Store previous batch's weights for transaction cost calculation
        prev_batch_weights = None
        
        with torch.no_grad():
            for batch_idx, (states, future_returns, masks) in enumerate(tqdm(self.test_loader, desc="Evaluating model")):
                states = states.to(self.device)
                future_returns = future_returns.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                portfolio_weights, winner_scores, sorted_indices = self.model(states, masks)
                
                # Calculate portfolio returns
                batch_size, T, num_assets = future_returns.shape
                
                # Store batch-level results
                all_portfolio_weights.append(portfolio_weights.cpu())
                all_winner_scores.append(winner_scores.cpu())
                
                # Process each batch element separately
                for b in range(batch_size):
                    # Episode level data
                    episode_weights = portfolio_weights[b].cpu().numpy()
                    
                    # Get portfolio returns for this sequence
                    portfolio_returns_timestep = self.calculate_returns(
                        portfolio_weights[b].unsqueeze(0),
                        future_returns[b].unsqueeze(0),
                        masks[b].unsqueeze(0)
                    ).squeeze(0)  # [T]
                    
                    # Store returns
                    episode_returns = portfolio_returns_timestep.cpu().numpy()
                    all_episode_returns.append(episode_returns)
                    all_episode_weights.append(episode_weights)
                    
                    # Visualize episode
                    if self.vis_manager and self.test_config.get("episode_visualizations", True):
                        if total_episodes < 20:  # Limit to first 20 episodes
                            self.vis_manager.visualize_episode(
                                episode_idx=total_episodes,
                                epoch=0,
                                batch_idx=batch_idx,
                                phase='test',
                                returns=episode_returns,
                                weights=episode_weights,
                                winner_scores=winner_scores[b].cpu().numpy()
                            )
                    
                    total_episodes += 1
                
                # Calculate batch returns
                portfolio_returns_batch = self.calculate_returns(
                    portfolio_weights,
                    future_returns,
                    masks
                )  # [batch_size, T]
                
                # Calculate transaction costs if needed
                if prev_batch_weights is not None and self.test_config.get("transaction_cost_analysis", True):
                    # Get the last weights from previous batch
                    prev_last_weights = prev_batch_weights[:, -1, :]  # [batch_size, num_assets]
                    
                    # Get the first weights from current batch
                    if batch_size > 0:  # Make sure we have data in this batch
                        current_first_weights = portfolio_weights[:, 0, :]  # [batch_size, num_assets]
                        
                        # Calculate weight changes
                        # We only process min(prev_batch_size, current_batch_size)
                        min_batch_size = min(prev_last_weights.size(0), current_first_weights.size(0))
                        
                        if min_batch_size > 0:
                            # Only use the common batch elements
                            weight_changes = torch.abs(
                                current_first_weights[:min_batch_size] - 
                                prev_last_weights[:min_batch_size]
                            )
                            
                            # Calculate transaction costs
                            transaction_costs = weight_changes.sum(dim=1) * self.transaction_cost_rate
                            
                            # Adjust first return of the current batch (only for common elements)
                            if portfolio_returns_batch.size(0) >= min_batch_size:
                                for b in range(min_batch_size):
                                    if b < portfolio_returns_batch.size(0):
                                        portfolio_returns_batch[b, 0] -= transaction_costs[b]
                
                # Store current weights for next iteration
                prev_batch_weights = portfolio_weights.detach().clone()
                
                all_portfolio_returns_step.append(portfolio_returns_batch.cpu())
                
                # Visualize batch
                if self.vis_manager and self.test_config.get("batch_visualizations", True):
                    if batch_idx % 5 == 0:  # Visualize every 5th batch
                        self.vis_manager.visualize_batch(
                            batch_idx=batch_idx,
                            epoch=0,
                            phase='test',
                            returns=portfolio_returns_batch.cpu().numpy(),
                            weights=portfolio_weights.cpu().numpy(),
                            sharpe_ratios=None,
                            winner_scores=winner_scores.cpu().numpy()
                        )
        
        # Concatenate all portfolio returns
        if all_portfolio_returns_step:
            all_returns = torch.cat(all_portfolio_returns_step, dim=0).numpy().flatten()
        else:
            all_returns = np.array([])
        
        # Concatenate all episode data
        all_episode_returns_array = np.array(all_episode_returns) if all_episode_returns else np.array([])
        all_episode_weights_array = np.array(all_episode_weights) if all_episode_weights else np.array([])
        
        # Calculate overall metrics
        metrics = self.calculate_performance_metrics(all_returns)
        
        # Calculate time-step specific metrics
        if len(all_episode_returns_array) > 0:  # Check if we have data
            timestep_metrics = self.calculate_timestep_metrics(all_episode_returns_array)
            metrics.update(timestep_metrics)
            
            # Perform bootstrap analysis
            bootstrap_metrics = self.perform_bootstrap_analysis(all_episode_returns_array)
            metrics.update(bootstrap_metrics)
            
            # Calculate portfolio composition metrics
            if len(all_episode_weights_array) > 0:  # Check if we have weights
                composition_metrics = self.calculate_composition_metrics(all_episode_weights_array)
                metrics.update(composition_metrics)
        
        # Compare against benchmark if needed
        if self.compare_benchmark and len(all_returns) > 0:
            benchmark_metrics = self.compare_to_benchmark(all_returns)
            metrics.update(benchmark_metrics)
        
        # Generate visualizations
        if len(all_returns) > 0:  # Check if we have data to visualize
            self.generate_visualizations(
                all_returns=all_returns,
                all_episode_returns=all_episode_returns_array,
                all_episode_weights=all_episode_weights_array,
                metrics=metrics
            )
        
        # Generate comprehensive report if needed
        # if self.test_config.get("generate_report", True):
        #     self.generate_report(metrics)
        
        return metrics
    
    def perform_bootstrap_analysis(self, episode_returns: np.ndarray) -> Dict[str, Any]:
        """
        Perform bootstrap analysis for confidence intervals.
        
        Args:
            episode_returns: Returns for each episode [num_episodes, T]
            
        Returns:
            Dictionary with bootstrap metrics
        """
        if episode_returns.shape[0] < 10 or self.bootstrap_iterations <= 0:
            return {'bootstrap_metrics': {}}
        
        num_episodes, T = episode_returns.shape
        
        # Initialize bootstrap results
        bootstrap_sharpes = []
        bootstrap_returns = []
        bootstrap_stds = []
        
        # Run bootstrap iterations
        for _ in range(self.bootstrap_iterations):
            # Sample episodes with replacement
            indices = np.random.choice(num_episodes, num_episodes, replace=True)
            sample = episode_returns[indices]
            
            # Flatten and calculate metrics
            sample_flat = sample.flatten()
            mean_return = np.mean(sample_flat)
            std_return = np.std(sample_flat)
            sharpe = mean_return / (std_return + 1e-8) * np.sqrt(12)
            
            bootstrap_returns.append(mean_return)
            bootstrap_stds.append(std_return)
            bootstrap_sharpes.append(sharpe)
        
        # Calculate confidence intervals (95%)
        ci_lower_idx = int(0.025 * self.bootstrap_iterations)
        ci_upper_idx = int(0.975 * self.bootstrap_iterations)
        
        sorted_returns = sorted(bootstrap_returns)
        sorted_sharpes = sorted(bootstrap_sharpes)
        
        bootstrap_metrics = {
            'sharpe_ci_lower': sorted_sharpes[ci_lower_idx],
            'sharpe_ci_upper': sorted_sharpes[ci_upper_idx],
            'return_ci_lower': sorted_returns[ci_lower_idx],
            'return_ci_upper': sorted_returns[ci_upper_idx],
            'bootstrap_iterations': self.bootstrap_iterations
        }
        
        return {'bootstrap_metrics': bootstrap_metrics}
    
    def calculate_composition_metrics(self, episode_weights: np.ndarray) -> Dict[str, Any]:
        """
        Calculate portfolio composition metrics.
        
        Args:
            episode_weights: Portfolio weights for each episode [num_episodes, T, num_assets]
            
        Returns:
            Dictionary with composition metrics
        """
        if episode_weights.shape[0] == 0:
            return {'composition_metrics': {}}
        
        num_episodes, T, num_assets = episode_weights.shape
        
        # Calculate long and short position statistics
        long_weights = np.maximum(episode_weights, 0)
        short_weights = np.minimum(episode_weights, 0)
        
        # Average number of assets used per time step
        avg_long_assets = np.mean(np.sum(long_weights > 0, axis=2))
        avg_short_assets = np.mean(np.sum(short_weights < 0, axis=2))
        
        # Average allocation to long and short positions
        avg_long_allocation = np.mean(np.sum(long_weights, axis=2))
        avg_short_allocation = np.mean(np.sum(short_weights, axis=2))
        
        # Asset utilization: how often each asset is used
        asset_utilization = np.mean(episode_weights != 0, axis=(0, 1))
        top_assets_idx = np.argsort(-asset_utilization)[:10]
        
        # Most commonly used assets
        top_assets = []
        if self.assets_list:
            top_assets = [self.assets_list[idx] for idx in top_assets_idx]
        
        # Calculate turnover (average weight change between time steps)
        turnovers = []
        for e in range(num_episodes):
            for t in range(1, T):
                prev_weights = episode_weights[e, t-1]
                curr_weights = episode_weights[e, t]
                turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2  # Divide by 2 to count only one side
                turnovers.append(turnover)
        
        avg_turnover = np.mean(turnovers) if turnovers else 0
        
        # Asset concentration (Gini coefficient)
        from scipy import stats
        abs_weights = np.abs(episode_weights)
        avg_abs_weights = np.mean(abs_weights, axis=(0, 1))
        
        # Sort weights for Gini calculation
        sorted_weights = np.sort(avg_abs_weights)
        n = len(sorted_weights)
        if n > 0:
            cumsum = np.cumsum(sorted_weights)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        else:
            gini = 0
        
        # Store as dictionary
        composition_metrics = {
            'avg_long_assets': float(avg_long_assets),
            'avg_short_assets': float(avg_short_assets),
            'avg_long_allocation': float(avg_long_allocation),
            'avg_short_allocation': float(avg_short_allocation),
            'avg_turnover': float(avg_turnover),
            'concentration_gini': float(gini),
            'top_assets_idx': top_assets_idx.tolist(),
            'top_assets': top_assets,
            'asset_utilization': asset_utilization.tolist()
        }
        
        return {'composition_metrics': composition_metrics}
    
    def compare_to_benchmark(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Compare strategy performance to benchmark. This is a stub function.
        In a real implementation, you would load benchmark data and calculate metrics.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dictionary with benchmark comparison metrics
        """
        # In a real implementation, load benchmark data here
        # For now, create a simple benchmark as normal returns with lower mean
        if len(returns) == 0:
            return {'benchmark_metrics': {}}
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Create synthetic benchmark returns
        benchmark_returns = np.random.normal(
            loc=mean_return * 0.7, 
            scale=std_return * 1.1,
            size=len(returns)
        )
        
        # Calculate benchmark metrics
        benchmark_mean = np.mean(benchmark_returns)
        benchmark_std = np.std(benchmark_returns)
        benchmark_sharpe = benchmark_mean / (benchmark_std + 1e-8) * np.sqrt(12)
        
        # Calculate strategy metrics
        strategy_mean = mean_return
        strategy_std = std_return
        strategy_sharpe = strategy_mean / (strategy_std + 1e-8) * np.sqrt(12)
        
        # Calculate alpha and beta
        # For simplicity, use a linear regression
        from scipy import stats
        beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_returns, returns)
        
        # Calculate information ratio
        tracking_error = np.std(returns - benchmark_returns)
        information_ratio = (strategy_mean - benchmark_mean) / (tracking_error + 1e-8)
        
        # Calculate outperformance statistics
        outperformance = returns - benchmark_returns
        pct_outperform = np.mean(outperformance > 0)
        
        # Store benchmark metrics
        benchmark_metrics = {
            'benchmark_mean_return': float(benchmark_mean),
            'benchmark_std_return': float(benchmark_std),
            'benchmark_sharpe': float(benchmark_sharpe),
            'strategy_alpha': float(alpha),
            'strategy_beta': float(beta),
            'r_squared': float(r_value**2),
            'information_ratio': float(information_ratio),
            'tracking_error': float(tracking_error),
            'percent_outperform': float(pct_outperform),
            'mean_outperformance': float(np.mean(outperformance))
        }
        
        return {'benchmark_metrics': benchmark_metrics}
    
    def generate_visualizations(self, all_returns: np.ndarray, all_episode_returns: np.ndarray,
                               all_episode_weights: np.ndarray, metrics: Dict[str, Any]):
        """
        Generate comprehensive visualizations for test results.
        
        Args:
            all_returns: Flattened returns
            all_episode_returns: Returns by episode [num_episodes, T]
            all_episode_weights: Weights by episode [num_episodes, T, num_assets]
            metrics: Dictionary of calculated metrics
        """
        if not self.test_config.get("generate_visualizations", True):
            return
        
        # Create visualization directory
        viz_dir = os.path.join(self.test_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Return distribution
        self._plot_return_distribution(all_returns, viz_dir)
        
        # 2. Cumulative returns
        self._plot_cumulative_returns(all_returns, viz_dir)
        
        # 3. Returns by time step
        self._plot_returns_by_timestep(all_episode_returns, viz_dir)
        
        # 4. Portfolio composition
        self._plot_portfolio_composition(all_episode_weights, viz_dir)
        
        # 5. Risk metrics
        self._plot_risk_metrics(all_returns, viz_dir)
        
        # 6. Benchmark comparison (if available)
        if 'benchmark_metrics' in metrics:
            self._plot_benchmark_comparison(all_returns, metrics['benchmark_metrics'], viz_dir)
        
        # 7. Portfolio turnover
        self._plot_portfolio_turnover(all_episode_weights, viz_dir)
        
        # 8. Combined performance dashboard
        self._plot_performance_dashboard(all_returns, all_episode_returns, metrics, viz_dir)
        
        logging.info(f"Generated test visualizations in {viz_dir}")
    
    def _plot_return_distribution(self, returns: np.ndarray, output_dir: str):
        """
        Plot distribution of returns.
        
        Args:
            returns: Array of returns
            output_dir: Output directory
        """
        plt.figure(figsize=(12, 8))
        
        # Plot histogram with KDE
        import seaborn as sns
        sns.histplot(returns * 100, kde=True, bins=30)
        
        # Add mean and zero lines
        mean_return = np.mean(returns) * 100
        plt.axvline(x=mean_return, color='red', linestyle='--', 
                  label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add annotations
        plt.title('Return Distribution', fontsize=14)
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Mean: {mean_return:.2f}%\n"
            f"Std Dev: {np.std(returns) * 100:.2f}%\n"
            f"Min: {np.min(returns) * 100:.2f}%\n"
            f"Max: {np.max(returns) * 100:.2f}%\n"
            f"Positive Returns: {np.mean(returns > 0) * 100:.1f}%"
        )
        plt.annotate(stats_text, xy=(0.02, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        # Save figure
        output_path = os.path.join(output_dir, "return_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cumulative_returns(self, returns: np.ndarray, output_dir: str):
        """
        Plot cumulative returns.
        
        Args:
            returns: Array of returns
            output_dir: Output directory
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod() - 1
        
        # Plot cumulative returns
        plt.plot(cum_returns * 100, 'b-', linewidth=2)
        
        # Add annotations
        plt.title('Cumulative Returns', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True)
        
        # Add final return annotation
        final_return = cum_returns[-1] * 100
        plt.annotate(f"Final: {final_return:.2f}%", 
                    xy=(len(cum_returns) - 1, final_return), 
                    xytext=(len(cum_returns) - 10, final_return + 2),
                    arrowprops=dict(arrowstyle='->'))
        
        # Calculate and plot drawdowns
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / (peak + 1)
        max_dd_idx = np.argmax(drawdown)
        max_dd = drawdown[max_dd_idx] * 100
        
        # Plot max drawdown
        dd_start = np.argmax(cum_returns[:max_dd_idx])
        plt.plot([dd_start, max_dd_idx], 
                [cum_returns[dd_start] * 100, cum_returns[max_dd_idx] * 100], 
                'r-', linewidth=2, alpha=0.7)
        
        plt.annotate(f"Max DD: {max_dd:.2f}%", 
                    xy=(max_dd_idx, cum_returns[max_dd_idx] * 100), 
                    xytext=(max_dd_idx - 10, cum_returns[max_dd_idx] * 100 - 5),
                    arrowprops=dict(arrowstyle='->'))
        
        # Save figure
        output_path = os.path.join(output_dir, "cumulative_returns.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_returns_by_timestep(self, episode_returns: np.ndarray, output_dir: str):
        """
        Plot returns by time step.
        
        Args:
            episode_returns: Returns by episode [num_episodes, T]
            output_dir: Output directory
        """
        if episode_returns.shape[0] == 0:
            return
        
        num_episodes, T = episode_returns.shape
        
        plt.figure(figsize=(14, 10))
        
        # Create a boxplot for each time step
        ax = plt.subplot(2, 1, 1)
        ax.boxplot(episode_returns * 100, notch=True, patch_artist=True,
                 boxprops=dict(facecolor='skyblue', alpha=0.7),
                 medianprops=dict(color='navy'),
                 flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
        
        ax.set_title('Return Distribution by Time Step', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticklabels([f"t={i+1}" for i in range(T)])
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Plot mean and confidence intervals
        ax2 = plt.subplot(2, 1, 2)
        x = np.arange(1, T + 1)
        
        # Calculate statistics for each time step
        mean_returns = np.mean(episode_returns, axis=0) * 100
        std_returns = np.std(episode_returns, axis=0) * 100
        
        # Plot mean line
        ax2.plot(x, mean_returns, 'b-', linewidth=2, label='Mean')
        
        # Add error bands
        ax2.fill_between(
            x,
            mean_returns - std_returns,
            mean_returns + std_returns,
            color='blue',
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        # Add zero line
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_title('Mean Return by Time Step with Variability', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        ax2.set_xticks(x)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "returns_by_timestep.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Cumulative returns by time step
        plt.figure(figsize=(12, 8))
        
        # Calculate cumulative returns for each episode
        cum_returns = np.cumprod(1 + episode_returns, axis=1) - 1
        
        # Plot individual paths
        for i in range(min(50, num_episodes)):
            plt.plot(x, cum_returns[i] * 100, 'b-', alpha=0.1)
        
        # Plot mean path
        mean_cum_returns = np.mean(cum_returns, axis=0) * 100
        plt.plot(x, mean_cum_returns, 'r-', linewidth=2, label='Mean')
        
        # Add annotations
        plt.title('Cumulative Returns by Time Step', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, "cumulative_returns_by_timestep.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_composition(self, episode_weights: np.ndarray, output_dir: str):
        """
        Plot portfolio composition.
        
        Args:
            episode_weights: Weights by episode [num_episodes, T, num_assets]
            output_dir: Output directory
        """
        if episode_weights.shape[0] == 0:
            return
        
        num_episodes, T, num_assets = episode_weights.shape
        
        # Calculate average weights across episodes
        avg_weights = np.mean(episode_weights, axis=0)  # [T, num_assets]
        
        # Find top assets by mean absolute weight
        mean_abs_weights = np.mean(np.abs(avg_weights), axis=0)
        top_indices = np.argsort(-mean_abs_weights)[:15]
        
        # Extract top assets names
        top_assets = []
        if self.assets_list:
            top_assets = [self.assets_list[idx] for idx in top_indices]
        else:
            top_assets = [f"Asset {idx+1}" for idx in top_indices]
        
        # Plot heatmap of top assets over time
        plt.figure(figsize=(14, 10))
        
        # Extract weights for top assets
        top_weights = avg_weights[:, top_indices]
        
        # Create heatmap
        import seaborn as sns
        sns.heatmap(
            top_weights.T,  # Transpose to have assets as rows
            cmap='RdBu_r',
            center=0,
            vmin=-np.max(np.abs(top_weights)),
            vmax=np.max(np.abs(top_weights)),
            yticklabels=top_assets,
            xticklabels=[f"t={i+1}" for i in range(T)],
            cbar_kws={'label': 'Average Portfolio Weight'}
        )
        
        plt.title('Portfolio Composition Over Time (Top 15 Assets by Weight)', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Asset', fontsize=12)
        
        # Save figure
        output_path = os.path.join(output_dir, "portfolio_composition.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Long vs. Short allocations over time
        plt.figure(figsize=(12, 8))
        
        # Calculate long and short allocations for each time step
        long_alloc = np.sum(np.maximum(avg_weights, 0), axis=1)
        short_alloc = np.sum(np.minimum(avg_weights, 0), axis=1)
        
        # Calculate number of long and short positions
        long_count = np.mean(np.sum(episode_weights > 0, axis=2), axis=0)
        short_count = np.mean(np.sum(episode_weights < 0, axis=2), axis=0)
        
        # Plot allocations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        x = np.arange(1, T + 1)
        
        # Plot allocations
        ax1.bar(x - 0.2, long_alloc, width=0.4, color='green', label='Long')
        ax1.bar(x + 0.2, -short_alloc, width=0.4, color='red', label='Short')
        
        ax1.set_title('Long vs. Short Allocations Over Time', fontsize=14)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Total Allocation', fontsize=12)
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.legend()
        ax1.set_xticks(x)
        
        # Plot position counts
        ax2.bar(x - 0.2, long_count, width=0.4, color='green', label='Long')
        ax2.bar(x + 0.2, short_count, width=0.4, color='red', label='Short')
        
        ax2.set_title('Number of Long vs. Short Positions Over Time', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Average Number of Positions', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.legend()
        ax2.set_xticks(x)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "long_short_allocation.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_metrics(self, returns: np.ndarray, output_dir: str):
        """
        Plot risk metrics.
        
        Args:
            returns: Array of returns
            output_dir: Output directory
        """
        if len(returns) == 0:
            return
        
        # Calculate various risk metrics
        from scipy import stats
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Return QQ plot
        ax1 = axes[0, 0]
        stats.probplot(returns, plot=ax1)
        ax1.set_title('Return QQ Plot', fontsize=14)
        
        # 2. Rolling volatility
        ax2 = axes[0, 1]
        window = min(30, len(returns) // 4)
        if window > 0:
            rolling_vol = pd.Series(returns).rolling(window=window).std() * np.sqrt(12) * 100
            ax2.plot(rolling_vol, 'b-')
            ax2.set_title(f'Rolling Volatility (Window={window})', fontsize=14)
            ax2.set_xlabel('Time Step', fontsize=12)
            ax2.set_ylabel('Annualized Volatility (%)', fontsize=12)
            ax2.grid(True)
        
        # 3. Return autocorrelation
        ax3 = axes[1, 0]
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(pd.Series(returns), ax=ax3)
        ax3.set_title('Return Autocorrelation', fontsize=14)
        
        # 4. Drawdown chart
        ax4 = axes[1, 1]
        cum_returns = (1 + returns).cumprod() - 1
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / (peak + 1) * 100
        
        ax4.plot(drawdown, 'r-')
        ax4.set_title('Drawdown Chart', fontsize=14)
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_ylabel('Drawdown (%)', fontsize=12)
        ax4.grid(True)
        
        # Add max drawdown annotation
        max_dd_idx = np.argmax(drawdown)
        max_dd = drawdown[max_dd_idx]
        ax4.annotate(f"Max DD: {max_dd:.2f}%", 
                    xy=(max_dd_idx, max_dd), 
                    xytext=(max_dd_idx - 10, max_dd - 5),
                    arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "risk_metrics.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_benchmark_comparison(self, returns: np.ndarray, benchmark_metrics: Dict[str, Any], output_dir: str):
        """
        Plot benchmark comparison.
        
        Args:
            returns: Strategy returns
            benchmark_metrics: Dictionary of benchmark metrics
            output_dir: Output directory
        """
        if 'benchmark_mean_return' not in benchmark_metrics:
            return
        
        # Create synthetic benchmark returns (in a real implementation, you would use actual benchmark data)
        benchmark_mean = benchmark_metrics['benchmark_mean_return']
        benchmark_std = benchmark_metrics['benchmark_std_return']
        np.random.seed(42)  # For reproducibility
        benchmark_returns = np.random.normal(loc=benchmark_mean, scale=benchmark_std, size=len(returns))
        
        # Calculate cumulative returns
        strategy_cum_returns = (1 + returns).cumprod() - 1
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        
        # Plot cumulative returns comparison
        plt.figure(figsize=(12, 8))
        
        plt.plot(strategy_cum_returns * 100, 'b-', linewidth=2, label='Strategy')
        plt.plot(benchmark_cum_returns * 100, 'r--', linewidth=2, label='Benchmark')
        
        plt.title('Strategy vs. Benchmark Cumulative Returns', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, "benchmark_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot rolling alpha
        plt.figure(figsize=(12, 8))
        
        window = min(30, len(returns) // 4)
        if window > 0:
            # Calculate rolling beta and alpha
            rolling_betas = []
            rolling_alphas = []
            
            for i in range(window, len(returns) + 1):
                x = benchmark_returns[i-window:i]
                y = returns[i-window:i]
                beta, alpha, _, _, _ = stats.linregress(x, y)
                rolling_betas.append(beta)
                rolling_alphas.append(alpha)
            
            # Plot rolling alpha
            plt.subplot(2, 1, 1)
            plt.plot(rolling_alphas, 'g-')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title(f'Rolling Alpha (Window={window})', fontsize=14)
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Alpha', fontsize=12)
            plt.grid(True)
            
            # Plot rolling beta
            plt.subplot(2, 1, 2)
            plt.plot(rolling_betas, 'b-')
            plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            plt.title(f'Rolling Beta (Window={window})', fontsize=14)
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Beta', fontsize=12)
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, "rolling_alpha_beta.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_portfolio_turnover(self, episode_weights: np.ndarray, output_dir: str):
        """
        Plot portfolio turnover.
        
        Args:
            episode_weights: Weights by episode [num_episodes, T, num_assets]
            output_dir: Output directory
        """
        if episode_weights.shape[0] == 0 or episode_weights.shape[1] <= 1:
            return
        
        num_episodes, T, num_assets = episode_weights.shape
        
        # Calculate turnover for each episode and time step
        turnovers = []
        for e in range(num_episodes):
            episode_turnover = []
            for t in range(1, T):
                prev_weights = episode_weights[e, t-1]
                curr_weights = episode_weights[e, t]
                turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2  # Divide by 2 to count only one side
                episode_turnover.append(turnover)
            turnovers.append(episode_turnover)
        
        # Convert to numpy array
        turnovers = np.array(turnovers)  # [num_episodes, T-1]
        
        # Calculate statistics
        mean_turnover = np.mean(turnovers, axis=0)
        std_turnover = np.std(turnovers, axis=0)
        
        # Plot turnover by time step
        plt.figure(figsize=(12, 8))
        
        x = np.arange(2, T + 1)  # Time steps 2 to T
        
        plt.plot(x, mean_turnover, 'b-', linewidth=2, label='Mean Turnover')
        plt.fill_between(
            x,
            mean_turnover - std_turnover,
            mean_turnover + std_turnover,
            color='blue',
            alpha=0.2,
            label='±1 Std Dev'
        )
        
        plt.title('Portfolio Turnover by Time Step', fontsize=14)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Turnover', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, "portfolio_turnover.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot turnover distribution
        plt.figure(figsize=(12, 8))
        
        sns.histplot(turnovers.flatten(), kde=True, bins=30)
        
        plt.axvline(x=np.mean(turnovers), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(turnovers):.4f}')
        
        plt.title('Turnover Distribution', fontsize=14)
        plt.xlabel('Turnover', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, "turnover_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_dashboard(self, returns: np.ndarray, episode_returns: np.ndarray, 
                                  metrics: Dict[str, Any], output_dir: str):
        """
        Plot comprehensive performance dashboard.
        
        Args:
            returns: Array of returns
            episode_returns: Returns by episode [num_episodes, T]
            metrics: Dictionary of metrics
            output_dir: Output directory
        """
        if len(returns) == 0:
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        gs = plt.GridSpec(7, 3, figure=fig)
        
        # 1. Cumulative returns
        cum_returns = (1 + returns).cumprod() - 1
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(cum_returns * 100, 'b-', linewidth=2)
        ax1.set_title('Cumulative Returns', fontsize=14)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.grid(True)
        
        # 2. Return distribution
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(returns * 100, kde=True, bins=30, ax=ax2)
        ax2.axvline(x=np.mean(returns) * 100, color='red', linestyle='--')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Return Distribution', fontsize=14)
        ax2.set_xlabel('Return (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        
        # 3. QQ plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(returns, plot=ax3)
        ax3.set_title('Return QQ Plot', fontsize=14)
        
        # 4. Drawdown chart
        ax4 = fig.add_subplot(gs[1, 2])
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / (peak + 1) * 100
        ax4.plot(drawdown, 'r-')
        ax4.set_title('Drawdown Chart', fontsize=14)
        ax4.set_xlabel('Time Step', fontsize=12)
        ax4.set_ylabel('Drawdown (%)', fontsize=12)
        ax4.grid(True)
        
        # 5. Returns by time step
        if episode_returns.shape[0] > 0:
            ax5 = fig.add_subplot(gs[2, :2])
            
            mean_returns = np.mean(episode_returns, axis=0) * 100
            std_returns = np.std(episode_returns, axis=0) * 100
            
            x = np.arange(1, episode_returns.shape[1] + 1)
            ax5.plot(x, mean_returns, 'b-', linewidth=2)
            ax5.fill_between(
                x,
                mean_returns - std_returns,
                mean_returns + std_returns,
                color='blue',
                alpha=0.2
            )
            
            ax5.set_title('Mean Return by Time Step', fontsize=14)
            ax5.set_xlabel('Time Step', fontsize=12)
            ax5.set_ylabel('Return (%)', fontsize=12)
            ax5.grid(True)
            ax5.set_xticks(x)
        
        # 6. Key metrics table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Extract metrics for the table
        metrics_text = "Performance Metrics\n-------------------\n"
        
        # Basic metrics
        metrics_text += f"Mean Return: {metrics.get('mean_return', 0) * 100:.2f}%\n"
        metrics_text += f"Std Deviation: {metrics.get('std_return', 0) * 100:.2f}%\n"
        metrics_text += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n"
        metrics_text += f"Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%\n"
        
        # More metrics
        if 'calmar_ratio' in metrics:
            metrics_text += f"Calmar Ratio: {metrics['calmar_ratio']:.4f}\n"
        if 'skewness' in metrics:
            metrics_text += f"Skewness: {metrics['skewness']:.4f}\n"
        if 'kurtosis' in metrics:
            metrics_text += f"Kurtosis: {metrics['kurtosis']:.4f}\n"
        if 'var_95' in metrics:
            metrics_text += f"95% VaR: {metrics['var_95'] * 100:.2f}%\n"
        if 'cvar_95' in metrics:
            metrics_text += f"95% CVaR: {metrics['cvar_95'] * 100:.2f}%\n"
        if 'percent_positive' in metrics:
            metrics_text += f"Positive Returns: {metrics['percent_positive'] * 100:.1f}%\n"
        
        # Add bootstrap metrics if available
        if 'bootstrap_metrics' in metrics and metrics['bootstrap_metrics']:
            bootstrap = metrics['bootstrap_metrics']
            metrics_text += "\nBootstrap Analysis\n-----------------\n"
            metrics_text += f"Sharpe CI: [{bootstrap.get('sharpe_ci_lower', 0):.4f}, {bootstrap.get('sharpe_ci_upper', 0):.4f}]\n"
            metrics_text += f"Return CI: [{bootstrap.get('return_ci_lower', 0) * 100:.2f}%, {bootstrap.get('return_ci_upper', 0) * 100:.2f}%]\n"
        
        # Add benchmark metrics if available
        if 'benchmark_metrics' in metrics and metrics['benchmark_metrics']:
            benchmark = metrics['benchmark_metrics']
            metrics_text += "\nBenchmark Comparison\n-------------------\n"
            metrics_text += f"Alpha: {benchmark.get('strategy_alpha', 0) * 100:.4f}%\n"
            metrics_text += f"Beta: {benchmark.get('strategy_beta', 0):.4f}\n"
            metrics_text += f"Info Ratio: {benchmark.get('information_ratio', 0):.4f}\n"
            metrics_text += f"Outperform %: {benchmark.get('percent_outperform', 0) * 100:.1f}%\n"
        
        # Add composition metrics if available
        if 'composition_metrics' in metrics and metrics['composition_metrics']:
            comp = metrics['composition_metrics']
            metrics_text += "\nPortfolio Composition\n--------------------\n"
            metrics_text += f"Avg Long Assets: {comp.get('avg_long_assets', 0):.1f}\n"
            metrics_text += f"Avg Short Assets: {comp.get('avg_short_assets', 0):.1f}\n"
            metrics_text += f"Avg Turnover: {comp.get('avg_turnover', 0):.4f}\n"
            metrics_text += f"Concentration: {comp.get('concentration_gini', 0):.4f}\n"
        
        ax6.text(0.05, 0.95, metrics_text, fontsize=12, va='top', family='monospace')
        
        # 7. Rolling metrics
        window = min(30, len(returns) // 4)
        if window > 0:
            # Rolling Sharpe
            ax7 = fig.add_subplot(gs[3, :])
            
            rolling_returns = pd.Series(returns).rolling(window=window).mean() * 12  # Annualized
            rolling_vol = pd.Series(returns).rolling(window=window).std() * np.sqrt(12)
            rolling_sharpe = rolling_returns / rolling_vol
            
            ax7.plot(rolling_sharpe, 'g-')
            ax7.axhline(y=metrics.get('sharpe_ratio', 0), color='red', linestyle='--',
                       label=f'Overall: {metrics.get("sharpe_ratio", 0):.4f}')
            
            ax7.set_title(f'Rolling Sharpe Ratio (Window={window})', fontsize=14)
            ax7.set_xlabel('Time Step', fontsize=12)
            ax7.set_ylabel('Sharpe Ratio', fontsize=12)
            ax7.grid(True)
            ax7.legend()
        
        # 8. Calendar analysis (if time information available)
        # This is a placeholder - in a real implementation, you would use actual dates
        ax8 = fig.add_subplot(gs[4, :])
        
        # Simulate monthly returns
        np.random.seed(42)
        monthly_returns = np.random.normal(
            loc=np.mean(returns),
            scale=np.std(returns),
            size=24
        )
        
        # Create a 2-year monthly calendar
        months_per_year = 12
        years = 2
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Reshape returns to calendar format
        calendar_returns = monthly_returns.reshape((years, months_per_year))
        
        # Create heatmap
        sns.heatmap(
            calendar_returns * 100,
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=.5,
            xticklabels=month_names,
            yticklabels=[f'Year {i+1}' for i in range(years)],
            ax=ax8,
            cbar_kws={'label': 'Monthly Return (%)'}
        )
        
        ax8.set_title('Monthly Returns Calendar (Simulated)', fontsize=14)
        
        # 9. Correlation matrix (if multiple assets or factors available)
        # This is a placeholder
        ax9 = fig.add_subplot(gs[5, :])
        
        # Simulate factor returns
        np.random.seed(42)
        num_factors = 5
        factor_names = ['Market', 'Size', 'Value', 'Momentum', 'Quality']
        factor_corr = np.array([
            [1.0, 0.2, -0.1, 0.3, 0.1],
            [0.2, 1.0, 0.1, 0.0, 0.2],
            [-0.1, 0.1, 1.0, -0.2, 0.3],
            [0.3, 0.0, -0.2, 1.0, 0.1],
            [0.1, 0.2, 0.3, 0.1, 1.0]
        ])
        
        # Create heatmap
        sns.heatmap(
            factor_corr,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=.5,
            xticklabels=factor_names,
            yticklabels=factor_names,
            ax=ax9,
            cbar_kws={'label': 'Correlation'}
        )
        
        ax9.set_title('Factor Correlation Matrix (Simulated)', fontsize=14)
        
        # 10. Additional analysis space
        ax10 = fig.add_subplot(gs[6, :])
        
        # Plot return statistics by quantile
        from matplotlib.ticker import PercentFormatter
        
        # Create return quantiles
        quantiles = 10
        returns_series = pd.Series(returns)
        returns_quantile = pd.qcut(returns_series, quantiles, labels=False)
        
        # Calculate statistics by quantile
        quantile_stats = pd.DataFrame({
            'returns': returns,
            'quantile': returns_quantile
        }).groupby('quantile').agg(
            mean_return=('returns', 'mean'),
            count=('returns', 'count'),
            std=('returns', 'std')
        )
        
        # Plot quantile returns
        ax10.bar(
            range(quantiles),
            quantile_stats['mean_return'] * 100,
            yerr=quantile_stats['std'] * 100,
            capsize=5,
            color='skyblue'
        )
        
        # Add counts
        for i, (_, row) in enumerate(quantile_stats.iterrows()):
            ax10.annotate(
                f"n={int(row['count'])}",
                xy=(i, row['mean_return'] * 100 + (row['std'] * 100 if row['mean_return'] >= 0 else -row['std'] * 100)),
                xytext=(0, 5 if row['mean_return'] >= 0 else -15),
                textcoords='offset points',
                ha='center'
            )
        
        ax10.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax10.set_title('Return Statistics by Quantile', fontsize=14)
        ax10.set_xlabel('Return Quantile (Low to High)', fontsize=12)
        ax10.set_ylabel('Mean Return (%)', fontsize=12)
        ax10.yaxis.set_major_formatter(PercentFormatter())
        ax10.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "performance_dashboard.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, metrics: Dict[str, Any]):
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Dictionary of metrics
        """
        report_dir = os.path.join(self.test_dir, "report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create report file
        report_path = os.path.join(report_dir, "performance_report.json")
        
        # Add timestamp and version info
        report_data = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "config": self.config.config
        }
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"Generated performance report at {report_path}")
        
        # Create HTML report (simple version)
        html_path = os.path.join(report_dir, "performance_report.html")
        
        # Basic HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlphaPortfolio Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .metrics-section {{ margin-bottom: 20px; }}
                .metric {{ margin: 5px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ margin-left: 10px; }}
                .image-gallery {{ display: flex; flex-wrap: wrap; }}
                .image-container {{ margin: 10px; max-width: 45%; }}
                .image-container img {{ max-width: 100%; }}
                .caption {{ text-align: center; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <h1>AlphaPortfolio Performance Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Performance Summary</h2>
            <div class="metrics-section">
                <div class="metric">
                    <span class="metric-name">Mean Return:</span>
                    <span class="metric-value">{metrics.get('mean_return', 0) * 100:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Sharpe Ratio:</span>
                    <span class="metric-value">{metrics.get('sharpe_ratio', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Maximum Drawdown:</span>
                    <span class="metric-value">{metrics.get('max_drawdown', 0) * 100:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Final Cumulative Return:</span>
                    <span class="metric-value">{metrics.get('final_cumulative_return', 0) * 100:.2f}%</span>
                </div>
            </div>
            
            <h2>Performance Visualizations</h2>
            <div class="image-gallery">
                <div class="image-container">
                    <img src="../visualizations/performance_dashboard.png" alt="Performance Dashboard">
                    <div class="caption">Performance Dashboard</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/cumulative_returns.png" alt="Cumulative Returns">
                    <div class="caption">Cumulative Returns</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/return_distribution.png" alt="Return Distribution">
                    <div class="caption">Return Distribution</div>
                </div>
                <div class="image-container">
                    <img src="../visualizations/portfolio_composition.png" alt="Portfolio Composition">
                    <div class="caption">Portfolio Composition</div>
                </div>
            </div>
            
            <h2>Full Metrics</h2>
            <pre id="full-metrics"></pre>
            <script>
                // Use a safe metrics rendering approach
                document.getElementById('full-metrics').textContent = ${JSON.stringify(metrics, null, 2)};
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Generated HTML performance report at {html_path}")
        
    # Additional safety fixes for TestEvaluator methods

    def calculate_performance_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics from returns with safety checks.
        
        Args:
            returns: Array of returns
                
        Returns:
            Dictionary with performance metrics
        """
        # Safety check for empty returns array
        if len(returns) == 0:
            logging.warning("Empty returns array provided to calculate_performance_metrics")
            return {
                'mean_return': 0.0,
                'std_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'percent_positive': 0.0,
                'max_consecutive_losses': 0,
                'final_cumulative_return': 0.0
            }
        
        try:
            # Basic statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / (std_return + 1e-8) * np.sqrt(12)  # Annualized
            
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod() - 1
            
            # Maximum drawdown
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (peak - cum_returns) / (peak + 1e-8)
            max_drawdown = np.max(drawdown)
            
            # Higher moments (skewness and kurtosis)
            try:
                from scipy import stats
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
            except:
                # Handle case where scipy might not be available
                skewness = 0.0
                kurtosis = 0.0
            
            # Calmar ratio (annualized return / max drawdown)
            calmar = (mean_return * 12) / (max_drawdown + 1e-8)
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)  # 95% Value at Risk
            cvar_95 = returns[returns <= var_95].mean() if any(returns <= var_95) else var_95  # 95% Conditional VaR
            
            # Positive returns statistics
            positive_returns = returns[returns > 0]
            pct_positive = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            
            # Maximum consecutive losses
            neg_streak = 0
            max_neg_streak = 0
            for ret in returns:
                if ret < 0:
                    neg_streak += 1
                    max_neg_streak = max(max_neg_streak, neg_streak)
                else:
                    neg_streak = 0
            
            metrics = {
                'mean_return': float(mean_return),
                'std_return': float(std_return),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'calmar_ratio': float(calmar),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'percent_positive': float(pct_positive),
                'max_consecutive_losses': int(max_neg_streak),
                'final_cumulative_return': float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0
            }
            
            return metrics
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {str(e)}")
            # Return default metrics
            return {
                'mean_return': 0.0,
                'std_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'percent_positive': 0.0,
                'max_consecutive_losses': 0,
                'final_cumulative_return': 0.0
            }

    def calculate_timestep_metrics(self, episode_returns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate metrics for each time step across episodes with safety checks.
        
        Args:
            episode_returns: Returns for each episode [num_episodes, T]
                
        Returns:
            Dictionary with time step metrics
        """
        # Safety check
        if episode_returns.shape[0] == 0 or len(episode_returns.shape) < 2:
            logging.warning("Invalid episode_returns shape in calculate_timestep_metrics")
            return {'timestep_metrics': {}}
        
        try:
            num_episodes, T = episode_returns.shape
            
            # Calculate statistics for each time step
            timestep_means = np.mean(episode_returns, axis=0)
            timestep_stds = np.std(episode_returns, axis=0)
            timestep_sharpes = timestep_means / (timestep_stds + 1e-8) * np.sqrt(12)
            
            # Calculate correlation matrix between time steps
            corr_matrix = np.corrcoef(episode_returns.T)
            
            # Calculate performance by time step
            pos_rates = np.mean(episode_returns > 0, axis=0)
            
            # Calculate performance trajectory
            cum_returns = np.cumprod(1 + episode_returns, axis=1) - 1
            avg_trajectory = np.mean(cum_returns, axis=0)
            
            # Store as dictionary
            timestep_metrics = {
                'mean_by_timestep': timestep_means.tolist(),
                'std_by_timestep': timestep_stds.tolist(),
                'sharpe_by_timestep': timestep_sharpes.tolist(),
                'correlation_matrix': corr_matrix.tolist(),
                'positive_rate_by_timestep': pos_rates.tolist(),
                'avg_cumulative_trajectory': avg_trajectory.tolist()
            }
            
            return {'timestep_metrics': timestep_metrics}
        except Exception as e:
            logging.error(f"Error calculating timestep metrics: {str(e)}")
            return {'timestep_metrics': {}}

    # Additional protection for the initialize_visualization method
    def initialize_visualization(self, num_assets: int, assets_list: Optional[List[str]] = None):
        """
        Initialize visualization manager with safety checks.
        
        Args:
            num_assets: Number of assets in the universe
            assets_list: Optional list of asset names
        """
        try:
            # Create default asset list if none provided
            if assets_list is None or len(assets_list) < num_assets:
                if assets_list is None:
                    assets_list = []
                    logging.warning(f"No asset list provided, creating default asset names")
                else:
                    logging.warning(f"Asset list length ({len(assets_list)}) is less than num_assets ({num_assets}), extending list")
                
                # Extend the list to match num_assets
                current_len = len(assets_list)
                assets_list.extend([f"Asset {i+1}" for i in range(current_len, num_assets)])
            
            # Create visualization manager
            self.vis_manager = VisualizationManager(
                output_dir=self.output_dir,
                cycle_idx="test",  # Use "test" as the cycle_idx for testing
                T=self.test_config.get("T", 12),  # Default to 12 if not specified
                num_assets=num_assets,
                assets_list=assets_list
            )
            
            logging.info("Visualization manager initialized for testing")
        except Exception as e:
            logging.error(f"Error initializing visualization: {str(e)}")
            self.vis_manager = None
