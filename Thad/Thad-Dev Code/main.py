from imports import *
from model_architecture import *
from training_model import *
from test_model import *
from data_pipeline import *
from plotting_functions_for_convergence import *
import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
lookback = 12
T = 12  # number of rebalancing steps per episode
model_G = 10  # number of assets selected for long/short in portfolio generation
batch_size = 1
num_epochs = 15
num_training_blocks = 4
test_year_start = 2015

# Instantiate data pipeline
data_pipeline = AlphaPortfolioData(lookback=lookback, T=T, num_training_blocks=num_training_blocks, test_year_start=test_year_start)
logger.info(f"Training rounds: {len(data_pipeline.train_sequences)}")
logger.info(f"Validation rounds: {len(data_pipeline.val_sequences)}")
logger.info(f"Test set episodes: {data_pipeline.test_sequences.shape[0]}")

# Create model
num_features = data_pipeline.num_features
model = AlphaPortfolioModel(num_features=num_features, lookback=lookback,
                            d_model=16, nhead=2, num_encoder_layers=1, d_attn=8, G=model_G)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Iterative training: for each round, train on its training set and validate on its validation set.
for round_idx in range(num_training_blocks):
    logger.info(f"Processing training round {round_idx}")

    # Create dataset for training round
    train_round_dataset = RoundDataset(data_pipeline.train_sequences[round_idx],
                                       data_pipeline.train_future_returns[round_idx],
                                       data_pipeline.train_masks[round_idx])
    val_round_dataset = RoundDataset(data_pipeline.val_sequences[round_idx],
                                     None,  # if validation future returns aren’t needed
                                     data_pipeline.val_masks[round_idx])
    
    # Train model on this round (train_model_sequential should accept a validation loader too)
    train_model_sequential(train_round_dataset, model, num_epochs=num_epochs, learning_rate=1e-4,
                           device=device, plots_dir='plots')
    evaluator = AlphaPortfolioEvaluator(val_round_dataset, model, device=device, G=model_G)
    model_sharpes = evaluator.test_model()         # Tests your deep RL model
    baseline_returns, baseline_sharpe = evaluator.evaluate_baseline()  # Baseline evaluation
    logger.info(f"Baseline Sharpe ratio on test set: {baseline_sharpe:.4f}")
    logger.info(f"Model Sharpe ratio on test set: {model_sharpes:.4f}")

# After training on all rounds, evaluate on the test set.
test_dataset = RoundDataset(data_pipeline.test_sequences, None, data_pipeline.test_masks)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
evaluator = AlphaPortfolioEvaluator(test_dataset, model, device = device, G = model_G)
model_sharpes = evaluator.test_model()         # Tests your deep RL model
baseline_returns, baseline_sharpe = evaluator.evaluate_baseline()  # Baseline evaluation
logger.info(f"Baseline Sharpe ratio on test set: {baseline_sharpe:.4f}")
logger.info(f"Model Sharpe ratio on test set: {model_sharpes:.4f}")