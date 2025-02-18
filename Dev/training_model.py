from imports import *
from model_architecture import *
from plotting_functions_for_convergence import *
from data_pipeline import *


# ------------------------- TRAINING PIPELINE -------------------------
def train_model_sequential(dataset, model, num_epochs=10, learning_rate=1e-4, device='cpu', batch_size=1, plots_dir='plots'):
    """
    Each episode consists of T sequential rebalancing steps.
    For each step t:
    - Get the state (shape: (num_assets, lookback, num_features))
    - Compute portfolio weights for that time step.
    - Get the one-month forward returns (shape: (num_assets,)) for that time step.
    - Compute the portfolio return (dot product) for that time step.
    After T steps, compute the Sharpe ratio of the T monthly returns as the delayed reward.
    """
    os.makedirs(plots_dir, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    all_epoch_avg_sharpes = []
    
    for epoch in range(num_epochs):
        logger.info(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
        epoch_sharpes = []
        for episode_idx, (state_seq, fwd_seq, mask_seq) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # state_seq: (B, T, A, lookback, num_features)
            # fwd_seq: (B, T, A)
            # mask_seq: (B, T, A)
            logger.info(f"[Episode {episode_idx}] state_seq shape: {state_seq.shape}, "
            f"fwd_seq shape: {fwd_seq.shape}, mask_seq shape: {mask_seq.shape}")
            
            B, T, A, L, F = state_seq.size()
            episode_episode_sharpes = [] # To store each episode's Sharpe ratio
            monthly_returns = []
            for t in range(T):
                state_t = state_seq[:, t, :, :, :] # (B, A, L, F)
                fwd_t = fwd_seq[:, t, :] # (B, A)
                mask_t = mask_seq[:, t, :] # (B, A)
                logger.info(f"[Episode {episode_idx}][Time {t}] state_t shape: {state_t.shape}, fwd_t shape: {fwd_t.shape}")
                
                portfolio_weights, winner_scores = model(state_t, mask_t)
                logger.info(f"[Episode {episode_idx}][Time {t}] portfolio_weights: {portfolio_weights}")
                logger.info(f"[Episode {episode_idx}][Time {t}] winner_scores: {winner_scores}")
                
                period_return = (portfolio_weights * fwd_t).sum(dim=1) # (B,)
                logger.info(f"[Episode {episode_idx}][Time {t}] period_return: {period_return}")
                monthly_returns.append(period_return.squeeze(0)) # assuming B=1
    
            monthly_returns = torch.stack(monthly_returns) # (T,)
            logger.info(f"[Episode {episode_idx}] Monthly returns: {monthly_returns}")
            
            mean_return = monthly_returns.mean()
            std_return = monthly_returns.std()
            sharpe_ratio = mean_return / (std_return + 1e-6)
            logger.info(f"[Episode {episode_idx}] Episode Sharpe Ratio: {sharpe_ratio}")
            
            loss = -sharpe_ratio
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_sharpes.append(sharpe_ratio.item())
            episode_episode_sharpes.append(sharpe_ratio.item())
            
    
        epoch_avg = np.mean(epoch_sharpes)
        all_epoch_avg_sharpes.append(epoch_avg)
        logger.info(f"Epoch {epoch+1}: Average Sharpe Ratio = {epoch_avg:.4f}")
    
    # After all epochs, plot the convergence of epoch-average Sharpe ratios.
    plot_epoch_path = os.path.join(plots_dir, "epoch_convergence.png")
    plot_epoch_sharpe(all_epoch_avg_sharpes, plot_epoch_path)
    logger.info("Training complete.") 