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
model_G = 5  # number of assets selected for long/short in portfolio generation
batch_size = 1
num_epochs = 15
num_training_blocks = 4
test_year_start = 2015
d_model = 64
nhead = 2
num_encoder_layers = 2
d_attn = 2
learning_rate = 1e-4

# Instantiate data pipeline
train_sequences = []
val_sequences = []
train_future_returns = []
val_future_returns = []
train_masks = []
val_masks = []
test_sequences = []
test_future_returns = []
test_masks = []

rerun_data_pipeline = False
if rerun_data_pipeline == True:
    data_pipeline = AlphaPortfolioData(lookback=lookback, T=T, num_training_blocks=num_training_blocks, test_year_start=test_year_start)
    
for file in range(int((count_files_in_directory("Data/Train/")) / 3)):
    train_sequences.append(torch.load(f"Data/Train/sequences_{file}.pt"))
    train_future_returns.append(torch.load(f"Data/Train/future_returns_{file}.pt"))
    train_masks.append(torch.load(f"Data/Train/masks_{file}.pt"))
for file in range(int((count_files_in_directory("Data/Val/")) / 3)):
    val_sequences.append(torch.load(f"Data/Val/sequences_{file}.pt"))
    val_future_returns.append(torch.load(f"Data/Val/future_returns_{file}.pt"))
    val_masks.append(torch.load(f"Data/Val/masks_{file}.pt"))
test_sequences = torch.load("Data/Test/sequences.pt")
test_future_returns = torch.load("Data/Test/future_returns.pt")
test_masks = torch.load("Data/Test/masks.pt")
    
logger.info(f"Training rounds: {len(train_sequences)}")
logger.info(f"Validation rounds: {len(val_sequences)}")
logger.info(f"Test set episodes: {test_sequences.shape[0]}")

# Create model
num_features = train_sequences[0].shape[-1]
model = AlphaPortfolioModel(num_features=num_features, lookback=lookback,
                            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, d_attn=d_attn, G=model_G)
# Check if a Metal-compatible GPU is available

if torch.backends.mps.is_available():
    device = torch.device("mps") # Use "mps" for Apple Metal devices (e.g., M1, M2)
    logger.info("Using Apple GPU with Metal backend.")
else:
    raise SystemExit("No compatible GPU found.")
model.to(device)
logger.info(f"Model device: {device}")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Iterative training: for each round, train on its training set and validate on its validation set.
for round_idx in range(num_training_blocks):
    logger.info(f"Processing training round {round_idx}")

    # Create dataset for training round
    train_round_dataset = RoundDataset(train_sequences[round_idx],
                                       train_future_returns[round_idx],
                                       train_masks[round_idx])
    val_round_dataset = RoundDataset(val_sequences[round_idx],
                                     val_future_returns[round_idx],
                                     val_masks[round_idx])
    
    
    # Train model on this round (train_model_sequential should accept a validation loader too)
    train_model_sequential(train_round_dataset, model, optimizer, num_epochs=num_epochs, plots_dir='plots', patience=5, device=device)
    evaluator = AlphaPortfolioEvaluator(val_round_dataset, model, device=device, G=model_G)
    model_sharpes = evaluator.test_model()         # Tests your deep RL model
    baseline_returns, baseline_sharpe = evaluator.evaluate_baseline()  # Baseline evaluation
    logger.info(f"Baseline Sharpe ratio on test set: {baseline_sharpe: .4f}")
    logger.info(f"Model Sharpe ratio on test set: {model_sharpes: .4f}")

# After training on all rounds, evaluate on the test set.
test_dataset = RoundDataset(test_sequences, test_future_returns, test_masks)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
evaluator = AlphaPortfolioEvaluator(test_dataset, model, device = device, G = model_G)
model_sharpes = evaluator.test_model()         # Tests your deep RL model
baseline_returns, baseline_sharpe = evaluator.evaluate_baseline()  # Baseline evaluation
logger.info(f"Baseline Sharpe ratio on test set: {baseline_sharpe: .4f}")
logger.info(f"Model Sharpe ratio on test set: {model_sharpes: .4f}")