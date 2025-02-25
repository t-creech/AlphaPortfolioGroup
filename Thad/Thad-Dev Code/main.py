from imports import *
from model_architecture import *
from training_model import *
from data_pipeline import *
from plotting_functions_for_convergence import *
import warnings
warnings.filterwarnings("ignore")

lookback = 12
start_year = 2017
final_year = 2018
end_year = 2020
T = 12 # number of rebalancing steps per episode
model_G = 10 # number of assets selected for long/short in portfolio generation
batch_size = 1
num_epochs = 15

dataset = AlphaPortfolioData(start_year=start_year, end_year=end_year, final_year=final_year, lookback=lookback, T=T)
logger.info(f"Dataset contains {dataset.sequences.shape[0]} episodes, each with {dataset.sequences.shape[1]} time steps, {dataset.sequences.shape[2]} assets.")

num_features = dataset.sequences.shape[-1]

model = AlphaPortfolioModel(num_features=num_features, lookback=lookback,
d_model=16, nhead=2, num_encoder_layers=1, d_attn=8, G=model_G)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

train_model_sequential(dataset, model, num_epochs=num_epochs, learning_rate=1e-4, device=device, batch_size=batch_size, plots_dir='plots')