# model.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from data import *

# model.py
# model.py
class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        logging.info(f"Initializing PositionalEncoding with d_model={d_model}, max_len={max_len}")
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:, :x.size(1)]

class SequenceRepresentationExtractionModel(nn.Module):
    """
    SREM using Transformer Encoder as described in the paper.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        logging.info(f"Initializing SREM with d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
        # Feature embedding
        self.feature_embedding = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        logging.info(f"SREM initialized with {sum(p.numel() for p in self.parameters())} parameters")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SREM.
        
        Args:
            x: Input tensor of shape [batch_size, num_assets, lookback, num_features]
               or [batch_size, T, num_assets, lookback, num_features]
            
        Returns:
            Asset representations of shape [batch_size, num_assets, lookback * d_model]
            or [batch_size, T, num_assets, lookback * d_model]
        """
        # Handle both 4D and 5D inputs
        if len(x.shape) == 5:
            # 5D input: [batch_size, T, num_assets, lookback, num_features]
            batch_size, T, num_assets, lookback, num_features = x.shape
            logging.debug(f"Processing 5D input with shape: {x.shape}")
            
            # Process each time step
            time_outputs = []
            for t in range(T):
                # Extract data for this time step
                t_data = x[:, t, :, :, :]  # [batch_size, num_assets, lookback, num_features]
                # Process this time step
                t_output = self._process_single_timestep(t_data)  # [batch_size, num_assets, lookback * d_model]
                time_outputs.append(t_output)
            
            # Stack outputs for all time steps
            return torch.stack(time_outputs, dim=1)  # [batch_size, T, num_assets, lookback * d_model]
            
        else:
            # 4D input: [batch_size, num_assets, lookback, num_features]
            logging.debug(f"Processing 4D input with shape: {x.shape}")
            return self._process_single_timestep(x)
    
    @profile_function
    def _process_single_timestep(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a single timestep with batch operations for improved efficiency.
        
        Args:
            x: Input tensor of shape [batch_size, num_assets, lookback, num_features]
            
        Returns:
            Asset representations of shape [batch_size, num_assets, lookback * d_model]
        """
        batch_size, num_assets, lookback, num_features = x.shape
        
        # Reshape to process all assets in parallel
        # Process in chunks to avoid memory issues with large models
        chunk_size = min(512, num_assets)
        asset_representations = []
        
        for i in range(0, num_assets, chunk_size):
            end_idx = min(i + chunk_size, num_assets)
            x_chunk = x[:, i:end_idx].reshape(-1, lookback, num_features)
            
            # Embed features for all assets at once
            embedded = self.feature_embedding(x_chunk)
            
            # Add positional encoding
            encoded = self.pos_encoder(embedded)
            
            # Apply transformer encoder to all assets at once
            encoded = self.transformer_encoder(encoded)
            
            # Reshape back to separate batch and assets
            chunk_size_actual = end_idx - i
            encoded = encoded.reshape(batch_size, chunk_size_actual, -1)
            asset_representations.append(encoded)
        
        # Concatenate all chunks
        return torch.cat(asset_representations, dim=1)
    
class CrossAssetAttentionNetwork(nn.Module):
    """
    Cross-Asset Attention Network (CAAN) as described in the paper.
    """
    def __init__(self, d_model: int):
        super().__init__()
        logging.info(f"Initializing CAAN with d_model={d_model}")
        
        # Projection matrices for query, key, and value
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        # Winner score projection
        self.W_s = nn.Linear(d_model, 1)
        
        # Scale factor for dot product attention
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float))
        
        logging.info(f"CAAN initialized with {sum(p.numel() for p in self.parameters())} parameters")
        
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CAAN.
        
        Args:
            r: Asset representations from SREM [batch_size, num_assets, d_model]
            
        Returns:
            Winner scores [batch_size, num_assets]
        """
        batch_size, num_assets, d_model = r.shape
        
        # Calculate query, key, and value for each asset
        Q = self.W_Q(r)  # [batch_size, num_assets, d_model]
        K = self.W_K(r)  # [batch_size, num_assets, d_model]
        V = self.W_V(r)  # [batch_size, num_assets, d_model]
        
        # Calculate attention scores
        # Compute dot products for all pairs
        attention_scores = torch.matmul(Q, K.transpose(1, 2))  # [batch_size, num_assets, num_assets]
        
        # Scale attention scores
        attention_scores = attention_scores / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_assets, num_assets]
        
        # Apply attention weights to values
        attention_output = torch.bmm(attention_weights, V)  # [batch_size, num_assets, d_model]
        
        # Calculate winner scores
        winner_scores = self.W_s(attention_output).squeeze(-1)  # [batch_size, num_assets]
        winner_scores = torch.tanh(winner_scores)  # Apply tanh as per the paper
        
        return winner_scores

class PortfolioGenerator(nn.Module):
    """
    Portfolio Generator as described in the paper.
    """
    def __init__(self, G: int = 20):
        """
        Initialize Portfolio Generator.
        
        Args:
            G: Number of assets to include in long and short portfolios
        """
        super().__init__()
        logging.info(f"Initializing PortfolioGenerator with G={G}")
        
        self.G = G
        
    def forward(self, winner_scores: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate portfolio weights based on winner scores.
        
        Args:
            winner_scores: Winner scores from CAAN [batch_size, num_assets]
            masks: Optional mask for valid assets [batch_size, num_assets]
            
        Returns:
            Tuple of (portfolio_weights, sorted_indices)
            - portfolio_weights: Portfolio weights [batch_size, num_assets]
            - sorted_indices: Indices of assets sorted by winner score [batch_size, num_assets]
        """
        batch_size, num_assets = winner_scores.shape
        
        # Apply mask if provided
        if masks is not None:
            # Set winner scores of invalid assets to negative infinity
            masked_scores = winner_scores.clone()
            masked_scores[masks == 0] = float('-inf')
        else:
            masked_scores = winner_scores
        
        # Sort assets by winner scores
        sorted_scores, sorted_indices = torch.sort(masked_scores, dim=1, descending=True)
        
        # Initialize portfolio weights
        portfolio_weights = torch.zeros_like(winner_scores)
        
        # Identify valid assets
        valid_count = num_assets
        if masks is not None:
            valid_count = masks.sum(dim=1).clamp(min=2*self.G)  # Ensure at least 2*G valid assets
        
        # Create long positions for top G assets and short positions for bottom G assets
        for b in range(batch_size):
            # Get number of valid assets for this batch
            G_adjusted = min(self.G, valid_count[b].item() // 2)
            
            # Extract top and bottom assets
            top_indices = sorted_indices[b, :G_adjusted]
            bottom_indices = sorted_indices[b, -G_adjusted:]
            
            # Get corresponding winner scores
            top_scores = winner_scores[b, top_indices]
            bottom_scores = -winner_scores[b, bottom_indices]  # Negative for short positions
            
            # Calculate weights using exponential normalization as described in the paper
            top_weights = torch.exp(top_scores)
            top_weights = top_weights / top_weights.sum()
            
            bottom_weights = torch.exp(bottom_scores)
            bottom_weights = bottom_weights / bottom_weights.sum()
            
            # Assign weights
            portfolio_weights[b, top_indices] = top_weights
            portfolio_weights[b, bottom_indices] = -bottom_weights  # Negative for short positions
        
        return portfolio_weights, sorted_indices

class AlphaPortfolio(nn.Module):
    """
    Complete AlphaPortfolio model as described in the paper.
    """
    def __init__(
        self,
        num_features: int,
        lookback: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        G: int = 20
    ):
        """
        Initialize AlphaPortfolio model.
        
        Args:
            num_features: Number of features per asset
            lookback: Number of lookback periods
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            G: Number of assets to include in long and short portfolios
        """
        super().__init__()
        logging.info(f"Initializing AlphaPortfolio with features={num_features}, lookback={lookback}, d_model={d_model}")
        
        # SREM
        self.srem = SequenceRepresentationExtractionModel(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        # CAAN - Note: d_model is multiplied by lookback because we concatenate all timesteps
        self.caan = CrossAssetAttentionNetwork(d_model * lookback)
        
        # Portfolio Generator
        self.portfolio_generator = PortfolioGenerator(G=G)
        
        # Store lookback for reference
        self.lookback = lookback
        
        # Total parameter count
        logging.info(f"AlphaPortfolio initialized with {sum(p.numel() for p in self.parameters())} parameters")
        
    @profile_function
    def forward(
        self, 
        states: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass for AlphaPortfolio.
        
        Args:
            states: State tensor [batch_size, num_assets, lookback, num_features]
                or [batch_size, T, num_assets, lookback, num_features]
            masks: Optional mask for valid assets [batch_size, num_assets]
                or [batch_size, T, num_assets]
            
        Returns:
            Tuple of (portfolio_weights, winner_scores, sorted_indices)
        """
        # Check if we're dealing with 5D input (includes time dimension)
        is_5d = len(states.shape) == 5
        
        if is_5d:
            batch_size, T, num_assets, lookback, num_features = states.shape
            
            # Process all time steps at once where possible
            states_reshaped = states.reshape(batch_size * T, num_assets, lookback, num_features)
            
            # Extract asset representations using SREM (more efficient)
            asset_representations = self.srem(states_reshaped)  # [batch_size*T, num_assets, lookback * d_model]
            
            # Reshape back to include time dimension
            asset_representations = asset_representations.reshape(batch_size, T, num_assets, -1)
            
            # Process each time step for CAAN
            all_weights = []
            all_scores = []
            all_indices = []
            
            for t in range(T):
                # Extract representations for this time step
                t_representations = asset_representations[:, t]  # [batch_size, num_assets, lookback * d_model]
                
                # Get masks for this time step if needed
                if masks is not None:
                    if len(masks.shape) == 3:  # Time-dependent masks
                        t_masks = masks[:, t]  # [batch_size, num_assets]
                    else:
                        t_masks = masks  # [batch_size, num_assets]
                else:
                    t_masks = None
                
                # Calculate winner scores and portfolio weights
                t_winner_scores = self.caan(t_representations)  # [batch_size, num_assets]
                t_weights, t_indices = self.portfolio_generator(t_winner_scores, t_masks)
                
                all_weights.append(t_weights)
                all_scores.append(t_winner_scores)
                all_indices.append(t_indices)
            
            # Stack results for all time steps
            portfolio_weights = torch.stack(all_weights, dim=1)  # [batch_size, T, num_assets]
            winner_scores = torch.stack(all_scores, dim=1)  # [batch_size, T, num_assets]
            sorted_indices = torch.stack(all_indices, dim=1)  # [batch_size, T, num_assets]
        else:
            # Standard 4D processing with optimizations
            asset_representations = self.srem(states)  # [batch_size, num_assets, lookback * d_model]
            winner_scores = self.caan(asset_representations)  # [batch_size, num_assets]
            portfolio_weights, sorted_indices = self.portfolio_generator(winner_scores, masks)
        
        return portfolio_weights, winner_scores, sorted_indices

