# main.py
import logging
import os
from typing import Callable
import yaml
import torch
import numpy as np
import random

import argparse
from datetime import datetime
from config import Config
from model import *
from data import *
# from enhanced_hyperparameter_search import *
from test import *
import logging
import os
# Import our custom modules
from config import Config
from enhanced_visualization import *
from enhanced_training import *
from model import *


def setup_logging(config):
    """Setup logging for the entire project."""
    log_dir = config.config["paths"]["log_dir"]
    log_level = config.config["logging"]["level"].upper()
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    log_filename = os.path.join(log_dir, f"alpha_portfolio_{config.config['experiment_id']}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level))
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level))
    
    # Reset root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure root logging
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial info
    logging.info(f"Logging initialized with level {log_level}")
    logging.info(f"Log file: {log_filename}")
    
    return log_filename

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def create_model(config, num_features):
    """Create AlphaPortfolio model from config."""
    model_config = config.config["model"]
    
    model = AlphaPortfolio(
        num_features=num_features,
        lookback=model_config["lookback"],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        G=model_config["G"]
    )
    
    return model

def generate_training_summary_plots(metrics: Dict[str, List[float]], output_dir: str, 
                                  cycle_idx: int, param_set_id: str):
    """Generate summary plots for all training epochs."""
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Sharpe ratio
    ax1 = axes[0, 0]
    epochs = range(1, len(metrics['train_sharpe']) + 1)
    ax1.plot(epochs, metrics['train_sharpe'], 'b-', linewidth=2, marker='o')
    
    # Add best epoch marker
    best_epoch = np.argmax(metrics['train_sharpe']) + 1
    best_sharpe = max(metrics['train_sharpe'])
    ax1.scatter([best_epoch], [best_sharpe], color='red', s=100, zorder=5)
    ax1.annotate(f"Best: {best_sharpe:.4f}", xy=(best_epoch, best_sharpe),
                xytext=(5, 5), textcoords='offset points')
    
    ax1.set_title('Training Sharpe Ratio by Epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.grid(True)
    
    # Plot loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['train_loss'], 'r-', linewidth=2, marker='o')
    
    # Add best epoch marker
    best_loss_epoch = np.argmin(metrics['train_loss']) + 1
    best_loss = min(metrics['train_loss'])
    ax2.scatter([best_loss_epoch], [best_loss], color='green', s=100, zorder=5)
    ax2.annotate(f"Best: {best_loss:.4f}", xy=(best_loss_epoch, best_loss),
                xytext=(5, 5), textcoords='offset points')
    
    ax2.set_title('Training Loss by Epoch', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True)
    
    # Plot returns
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics['train_returns'], 'g-', linewidth=2, marker='o')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_title('Mean Returns by Epoch', fontsize=14)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Mean Return', fontsize=12)
    ax3.grid(True)
    
    # Plot std
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics['train_std'], 'm-', linewidth=2, marker='o')
    
    ax4.set_title('Return Standard Deviation by Epoch', fontsize=14)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.grid(True)
    
    plt.suptitle(f'Training Summary for Cycle {cycle_idx}, Params {param_set_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(output_dir, f"training_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_train_val_comparison_plots(metrics: Dict[str, List[float]], output_dir: str, 
                                      cycle_idx: int, param_set_id: str):
    """Generate plots comparing training and validation performance."""
    # Check if validation metrics exist
    if not metrics['val_sharpe']:
        logging.warning("No validation metrics available for comparison plot")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get epochs for x-axis
    epochs = range(1, len(metrics['train_sharpe']) + 1)
    
    # Plot Sharpe ratio
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics['train_sharpe'], 'b-', linewidth=2, label='Training')
    
    # Add validation point
    val_sharpe = metrics['val_sharpe'][-1]
    ax1.axhline(y=val_sharpe, color='r', linestyle='--', 
               label=f'Validation: {val_sharpe:.4f}')
    
    ax1.set_title('Sharpe Ratio', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.grid(True)
    ax1.legend()
    
    # Plot loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['train_loss'], 'b-', linewidth=2, label='Training')
    
    # Add validation point
    val_loss = metrics['val_loss'][-1]
    ax2.axhline(y=val_loss, color='r', linestyle='--',
               label=f'Validation: {val_loss:.4f}')
    
    ax2.set_title('Loss', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True)
    ax2.legend()
    
    # Plot returns
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics['train_returns'], 'b-', linewidth=2, label='Training')
    
    # Add validation point
    val_returns = metrics['val_returns'][-1]
    ax3.axhline(y=val_returns, color='r', linestyle='--',
               label=f'Validation: {val_returns:.4f}')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_title('Mean Returns', fontsize=14)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Mean Return', fontsize=12)
    ax3.grid(True)
    ax3.legend()
    
    # Plot std
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics['train_std'], 'b-', linewidth=2, label='Training')
    
    # Add validation point
    val_std = metrics['val_std'][-1]
    ax4.axhline(y=val_std, color='r', linestyle='--',
               label=f'Validation: {val_std:.4f}')
    
    ax4.set_title('Return Standard Deviation', fontsize=14)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.grid(True)
    ax4.legend()
    
    plt.suptitle(f'Training vs Validation Performance for Cycle {cycle_idx}, Params {param_set_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(output_dir, f"train_val_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@profile_function
def train_cycle(
    model: nn.Module,
    train_loader,
    val_loader,
    config,
    cycle_params: Dict[str, Any],
    epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    param_set_id: Optional[str] = None  # Added param_set_id parameter
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model for one cycle, using unique directories for each parameter set.
    
    Args:
        model: AlphaPortfolio model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        cycle_params: Parameters for current cycle
        epoch_callback: Optional callback function to track epoch metrics
        param_set_id: Optional identifier for the parameter set (for hyperparameter search)
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Extract parameters
    cycle_idx = cycle_params["cycle_idx"]
    num_epochs = config.config["training"]["num_epochs"]
    lr = config.config["training"]["learning_rate"]
    weight_decay = config.config["training"]["weight_decay"]
    output_dir = config.config["paths"]["plot_dir"]
    model_dir = config.config["paths"]["model_dir"]
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Use "mps" for Apple Metal devices (e.g., M1, M2)
        logging.info("Using Apple GPU with Metal backend.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
    
    # Create unique directory identifier based on hyperparameters if not provided
    if param_set_id is None:
        # Create a concise identifier from the model parameters
        d_model = config.config["model"]["d_model"]
        nhead = config.config["model"]["nhead"]
        num_layers = config.config["model"]["num_layers"]
        G = config.config["model"]["G"]
        param_set_id = f"d{d_model}_h{nhead}_l{num_layers}_G{G}_lr{lr}_wd{weight_decay}"
    
    # Create necessary directories with unique parameter set folder
    param_dir = os.path.join(output_dir, f"cycle_{cycle_idx}", f"params_{param_set_id}")
    summary_dir = os.path.join(param_dir, "summary")
    model_param_dir = os.path.join(model_dir, f"cycle_{cycle_idx}", f"params_{param_set_id}")
    
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(model_param_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = RLTrainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
    
    # Initialize metrics storage
    metrics = {
        'train_loss': [],
        'train_sharpe': [],
        'train_returns': [],
        'train_std': [],
        'val_loss': [],
        'val_sharpe': [],
        'val_returns': [],
        'val_std': []
    }
    
    # Training loop
    logging.info(f"Starting training for Cycle {cycle_idx}, Parameter Set {param_set_id}")
    logging.info(f"  Training period: {cycle_params['train_start']} to {cycle_params['train_end']}")
    logging.info(f"  Validation will be performed at the end using data from: {cycle_params['validate_start']} to {cycle_params['validate_end']}")
    
    # Track best epoch for saving model
    best_train_sharpe = -float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch with parameter-specific output directory
        train_metrics = trainer.train_epoch(
            train_loader=train_loader,
            epoch=epoch,
            output_dir=param_dir,  # Use parameter-specific directory
            cycle_idx=cycle_idx,
            param_set_id=param_set_id  # Pass parameter set ID
        )
        
        # Update metrics
        metrics['train_loss'].append(train_metrics.get('loss', float('inf')))
        metrics['train_sharpe'].append(train_metrics.get('sharpe_ratio', -float('inf')))
        metrics['train_returns'].append(train_metrics.get('mean_return', 0.0))
        metrics['train_std'].append(train_metrics.get('std_return', 0.0))
        
        # Check if this is the best epoch so far
        current_sharpe = train_metrics.get('sharpe_ratio', -float('inf'))
        if current_sharpe > best_train_sharpe:
            best_train_sharpe = current_sharpe
            best_epoch = epoch
            
            # Save best model during training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'train_sharpe': best_train_sharpe,
                'param_set_id': param_set_id  # Include parameter set ID
            }, os.path.join(model_param_dir, f"best_train.pt"))
            logging.info(f"  Saved best training model at epoch {epoch} with Sharpe: {best_train_sharpe:.4f}")
        
        # Call epoch callback if provided
        if epoch_callback is not None:
            epoch_metrics = {
                'train_loss': train_metrics.get('loss', float('inf')),
                'train_sharpe': train_metrics.get('sharpe_ratio', -float('inf')),
                'train_returns': train_metrics.get('mean_return', 0.0),
                'train_std': train_metrics.get('std_return', 0.0)
            }
            epoch_callback(epoch, epoch_metrics)
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'param_set_id': param_set_id  # Include parameter set ID
            }, os.path.join(model_param_dir, f"epoch_{epoch}.pt"))
            logging.info(f"  Saved checkpoint at epoch {epoch}")
        
        # Log epoch summary
        epoch_time = time.time() - start_time
        logging.info(f"  Epoch {epoch} completed in {epoch_time:.2f}s, "
                   f"Train Sharpe: {train_metrics.get('sharpe_ratio', -float('inf')):.4f}")
    
    # Generate summary training plots after all epochs
    generate_training_summary_plots(metrics, summary_dir, cycle_idx, param_set_id)
    logging.info(f"Generated summary training plots in {summary_dir}")
    
    # Validate once after all epochs with detailed visualization
    logging.info(f"Validating model after completing all epochs (param set {param_set_id})")
    val_metrics = trainer.validate_comprehensive(
        val_loader=val_loader,
        epoch=num_epochs-1,
        output_dir=param_dir,  # Use parameter-specific directory
        cycle_idx=cycle_idx,
        treat_as_test=True,  # This flag enables more detailed visualization
        param_set_id=param_set_id  # Pass parameter set ID
    )
    
    # Update validation metrics
    metrics['val_loss'].append(val_metrics.get('loss', float('inf')))
    metrics['val_sharpe'].append(val_metrics.get('sharpe_ratio', -float('inf')))
    metrics['val_returns'].append(val_metrics.get('mean_return', 0.0))
    metrics['val_std'].append(val_metrics.get('std_return', 0.0))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'train_sharpe': best_train_sharpe,
        'train_epoch': best_epoch,
        'val_sharpe': val_metrics.get('sharpe_ratio', -float('inf')),
        'param_set_id': param_set_id,  # Include parameter set ID
        'hyperparameters': {
            'd_model': config.config["model"]["d_model"],
            'nhead': config.config["model"]["nhead"],
            'num_layers': config.config["model"]["num_layers"],
            'G': config.config["model"]["G"],
            'learning_rate': lr,
            'weight_decay': weight_decay
        }
    }, os.path.join(model_param_dir, "final.pt"))
    logging.info(f"Saved final model for cycle {cycle_idx}, param set {param_set_id}")
    
    # Create a combined plot with training and validation
    generate_train_val_comparison_plots(metrics, summary_dir, cycle_idx, param_set_id)
    logging.info(f"Generated training-validation comparison plots in {summary_dir}")
    
    logging.info(f"Cycle {cycle_idx}, param set {param_set_id} training completed with validation Sharpe: {val_metrics.get('sharpe_ratio', -float('inf')):.4f}")
    
    return model, metrics

@profile_function
def train_model(
    model: nn.Module,
    train_loader,
    config,
    cycle_params: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Simplified function to train the model without validation.
    
    Args:
        model: AlphaPortfolio model
        train_loader: Training data loader
        config: Configuration object
        cycle_params: Parameters for current cycle
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Extract parameters
    cycle_idx = cycle_params["cycle_idx"]
    num_epochs = config.config["training"]["num_epochs"]
    lr = float(config.config["training"]["learning_rate"])
    weight_decay = float(config.config["training"]["weight_decay"])
    output_dir = config.config["paths"]["plot_dir"]
    model_dir = config.config["paths"]["model_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a concise identifier from the model parameters
    d_model = config.config["model"]["d_model"]
    nhead = config.config["model"]["nhead"]
    num_layers = config.config["model"]["num_layers"]
    G = config.config["model"]["G"]
    param_set_id = f"d{d_model}_h{nhead}_l{num_layers}_G{G}_lr{lr}_wd{weight_decay}"
    
    # Create necessary directories
    param_dir = os.path.join(output_dir, f"cycle_{cycle_idx}", f"params_{param_set_id}")
    summary_dir = os.path.join(param_dir, "summary")
    model_param_dir = os.path.join(model_dir, f"cycle_{cycle_idx}", f"params_{param_set_id}")
    
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(model_param_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = RLTrainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
    
    # Initialize metrics storage
    metrics = {
        'train_loss': [],
        'train_sharpe': [],
        'train_returns': [],
        'train_std': []
    }
    
    # Training loop
    logging.info(f"Starting training for Cycle {cycle_idx}")
    logging.info(f"  Training period: {cycle_params['train_start']} to {cycle_params['train_end']}")
    
    # Track best epoch for saving model
    best_train_sharpe = -float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(
            train_loader=train_loader,
            epoch=epoch,
            output_dir=param_dir,
            cycle_idx=cycle_idx,
            param_set_id=param_set_id
        )
        
        # Update metrics
        metrics['train_loss'].append(train_metrics.get('loss', float('inf')))
        metrics['train_sharpe'].append(train_metrics.get('sharpe_ratio', -float('inf')))
        metrics['train_returns'].append(train_metrics.get('mean_return', 0.0))
        metrics['train_std'].append(train_metrics.get('std_return', 0.0))
        
        # Check if this is the best epoch so far
        current_sharpe = train_metrics.get('sharpe_ratio', -float('inf'))
        if current_sharpe > best_train_sharpe:
            best_train_sharpe = current_sharpe
            best_epoch = epoch
            
            # Save best model during training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'train_sharpe': best_train_sharpe,
                'param_set_id': param_set_id
            }, os.path.join(model_param_dir, f"best_train.pt"))
            logging.info(f"  Saved best training model at epoch {epoch} with Sharpe: {best_train_sharpe:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'param_set_id': param_set_id
            }, os.path.join(model_param_dir, f"epoch_{epoch}.pt"))
            logging.info(f"  Saved checkpoint at epoch {epoch}")
        
        # Log epoch summary
        epoch_time = time.time() - start_time
        logging.info(f"  Epoch {epoch} completed in {epoch_time:.2f}s, "
                   f"Train Sharpe: {train_metrics.get('sharpe_ratio', -float('inf')):.4f}")
    
    # Generate summary training plots after all epochs
    generate_training_summary_plots(metrics, summary_dir, cycle_idx, param_set_id)
    logging.info(f"Generated summary training plots in {summary_dir}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'train_sharpe': best_train_sharpe,
        'train_epoch': best_epoch,
        'param_set_id': param_set_id,
        'hyperparameters': {
            'd_model': config.config["model"]["d_model"],
            'nhead': config.config["model"]["nhead"],
            'num_layers': config.config["model"]["num_layers"],
            'G': config.config["model"]["G"],
            'learning_rate': lr,
            'weight_decay': weight_decay
        }
    }, os.path.join(model_param_dir, "final.pt"))
    logging.info(f"Saved final model for cycle {cycle_idx}, param set {param_set_id}")
    
    # Load best model
    best_model_path = os.path.join(model_param_dir, f"best_train.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded best model from epoch {checkpoint['epoch']} with training Sharpe: {checkpoint['train_sharpe']:.4f}")
    
    logging.info(f"Cycle {cycle_idx} training completed")
    
    return model, metrics

@profile_function
def test_model(model, test_loader, config, scaler_path=None):
    """
    Test the model using TestEvaluator for comprehensive evaluation and plots.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config: Configuration object
        scaler_path: Optional path to scaler used in training
        
    Returns:
        Dictionary with test metrics
    """
    logging.info("Starting model testing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get some test config parameters
    test_output_dir = os.path.join(config.config["paths"]["output_dir"], "test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Initialize test evaluator
    test_evaluator = TestEvaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Run evaluation
    test_metrics = test_evaluator.evaluate()
    
    # Save test results
    test_results = {
        'metrics': test_metrics,
        'hyperparameters': {
            'd_model': config.config["model"]["d_model"],
            'nhead': config.config["model"]["nhead"],
            'num_layers': config.config["model"]["num_layers"],
            'G': config.config["model"]["G"],
            'learning_rate': config.config["training"]["learning_rate"],
            'weight_decay': config.config["training"]["weight_decay"]
        },
        'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results to JSON
    import json
    with open(os.path.join(test_output_dir, "test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logging.info(f"Testing completed with Sharpe ratio: {test_metrics.get('sharpe_ratio', -float('inf')):.4f}")
    return test_metrics

def main():
    """
    Main function to run AlphaPortfolio training with simplified workflow.
    
    Workflow:
    1. Train on specified cycle dates with predefined parameters
    2. Test the trained model with comprehensive visualization
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AlphaPortfolio Training (Simplified)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'full'],
                        default='full', help='Mode of operation (train only, test only, or both)')
    parser.add_argument('--cycle', type=int, default=0, help='Cycle index to run')
    parser.add_argument('--output', type=str, default=None, help='Custom output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override output directory if specified
    if args.output:
        config.config["paths"]["output_dir"] = args.output
        config.config["paths"]["model_dir"] = os.path.join(args.output, "models")
        config.config["paths"]["log_dir"] = os.path.join(args.output, "logs")
        config.config["paths"]["plot_dir"] = os.path.join(args.output, "plots")
        config._create_directories()
    
    # Setup logging
    setup_logging(config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Log start
    logging.info(f"Starting AlphaPortfolio experiment: {config.config['experiment_id']}")
    logging.info(f"Mode: {args.mode}")
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get data path
    data_path = config.config["paths"]["data_path"]
    logging.info(f"Using data from: {data_path}")
    
    # Get cycles from config
    all_cycles = config.get_all_cycles()
    if not all_cycles:
        # If no cycles defined, create a default one
        default_cycle = {
            "cycle_idx": 0,
            "train_start": "2010-01-01",
            "train_end": "2014-12-31",
            "test_start": "2015-01-01",
            "test_end": "2020-12-31"
        }
        all_cycles = [default_cycle]
    
    # Select cycle to run
    cycle_to_run = next((c for c in all_cycles if c["cycle_idx"] == args.cycle), all_cycles[0])
    logging.info(f"Running cycle {cycle_to_run['cycle_idx']}")
    
    #######################
    # TRAINING PHASE
    #######################
    trained_model = None
    scaler_path = None
    
    if args.mode == 'train' or args.mode == 'full':
        logging.info("=== Starting Training Phase ===")
        
        # Get training parameters
        train_start = cycle_to_run.get("train_start")
        train_end = cycle_to_run.get("train_end")
        
        logging.info(f"Training period: {train_start} to {train_end}")
        
        # Create training data
        train_data = AlphaPortfolioData(
            data_path=data_path,
            start_date=train_start,
            end_date=train_end,
            T=config.config["model"]["T"],
            lookback=config.config["model"]["lookback"],
            cycle_id=cycle_to_run["cycle_idx"],
            experiment_id=config.config["experiment_id"]
        )
        
        # Create training loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.config["training"]["batch_size"],
            shuffle=True,
            num_workers=config.config["training"]["num_workers"],
            pin_memory=True
        )
        
        # Get feature count
        num_features = train_data.get_feature_count()
        
        # Create model
        model = create_model(config, num_features)
        model = model.to(device)
        
        # Train model
        trained_model, metrics = train_model(
            model=model,
            train_loader=train_loader,
            config=config,
            cycle_params=cycle_to_run
        )
        
        # Save scaler path for testing
        scaler_path = train_data.get_scaler_path()
        
    #######################
    # TESTING PHASE
    #######################
    if args.mode == 'test' or args.mode == 'full':
        logging.info("=== Starting Testing Phase ===")
        
        # If we're only testing and haven't trained a model
        if trained_model is None:
            # Load the model from the saved best model
            model_dir = config.config["paths"]["model_dir"]
            model_path = os.path.join(model_dir, f"cycle_{cycle_to_run['cycle_idx']}", "final.pt")
            
            if not os.path.exists(model_path):
                # Try to find any model file
                import glob
                model_files = glob.glob(os.path.join(model_dir, f"cycle_{cycle_to_run['cycle_idx']}", "**/*.pt"), 
                                         recursive=True)
                if model_files:
                    model_path = model_files[0]
                    logging.info(f"Found model file: {model_path}")
                else:
                    logging.error(f"No model file found for cycle {cycle_to_run['cycle_idx']}. Cannot test.")
                    return
            
            # Load model
            checkpoint = torch.load(model_path)
            
            # Get model parameters
            num_features = train_data.get_feature_count() if 'train_data' in locals() else 100  # Default if unknown
            
            # Create model
            model = create_model(config, num_features)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            trained_model = model
            
            logging.info(f"Loaded model from {model_path}")
        
        # Get test parameters
        test_start = cycle_to_run.get("test_start", "2015-01-01")
        test_end = cycle_to_run.get("test_end", "2020-12-31")
        
        logging.info(f"Test period: {test_start} to {test_end}")
        
        # Create test data
        test_data = AlphaPortfolioData(
            data_path=config.config["test"].get("data_path", data_path),
            start_date=test_start,
            end_date=test_end,
            T=config.config["model"]["T"],
            lookback=config.config["model"]["lookback"],
            cycle_id=999,  # Special ID for test
            experiment_id=config.config["experiment_id"],
            scaler_path=scaler_path
        )
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=config.config["training"]["batch_size"],
            shuffle=False,
            num_workers=config.config["training"]["num_workers"],
            pin_memory=True
        )
        
        # Test model
        test_metrics = test_model(
            model=trained_model,
            test_loader=test_loader,
            config=config,
            scaler_path=scaler_path
        )
    
    logging.info("AlphaPortfolio experiment completed")

if __name__ == "__main__":
    main()