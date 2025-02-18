from imports import *

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