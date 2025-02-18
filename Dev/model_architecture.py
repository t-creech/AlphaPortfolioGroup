from imports import *

# ------------------------- MODEL ARCHITECTURE -------------------------
class AlphaPortfolioModel(nn.Module):
    def __init__(self, num_features, lookback, d_model=32, nhead=4, num_encoder_layers=2, d_attn=16, G=5):
        """
        Processes an asset's historical state (shape: (num_assets, lookback, num_features))
        and produces a portfolio weight vector (one weight per asset).
        Uses:
          - Input projection (Linear) to embed each time step.
          - A Transformer encoder (SREM) for temporal dependencies.
          - A CAAN module for inter-asset relationships.
          - A portfolio generator that selects top and bottom G assets.
        """
        super().__init__()
        self.G = G
        self.lookback = lookback
        self.d_model = d_model

        logger.info(f"Initializing AlphaPortfolioModel with num_features={num_features}, lookback={lookback}, "
                    f"d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}, "
                    f"d_attn={d_attn}, G={G}")
        
        self.input_projection = nn.Linear(num_features, d_model)
        logger.info(f"Input projection layer: {self.input_projection}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        logger.info("Transformer encoder initialized.")

        self.r_dim = lookback * d_model
        logger.info(f"Asset representation dimension (r_dim): {self.r_dim}")

        self.query_layer = nn.Linear(self.r_dim, d_attn)
        self.key_layer = nn.Linear(self.r_dim, d_attn)
        self.value_layer = nn.Linear(self.r_dim, d_attn)
        logger.info("CAAN layers (query, key, value) initialized.")

        self.score_layer = nn.Linear(d_attn, 1)
        logger.info("Score layer for winner scores initialized.")

    def forward(self, x, mask):
        logger.info(f"[Model] Input x shape: {x.shape}")
        B, A, L, feat_dim = x.size()
        x = x.view(B * A, L, feat_dim)
        logger.info(f"[Model] After flattening: {x.shape}")
        
        x = self.input_projection(x)
        logger.info(f"[Model] After input projection: {x.shape}")
        
        x = x.transpose(0, 1)
        logger.info(f"[Model] After transpose for Transformer: {x.shape}")
        
        encoded = self.transformer_encoder(x)
        logger.info(f"[Model] After Transformer encoder: {encoded.shape}")
        
        encoded = encoded.transpose(0, 1)
        logger.info(f"[Model] After transpose back: {encoded.shape}")
        
        asset_repr = encoded.contiguous().view(B, A, -1)
        logger.info(f"[Model] Asset representation shape: {asset_repr.shape}")
        
        Q = self.query_layer(asset_repr)
        K = self.key_layer(asset_repr)
        V = self.value_layer(asset_repr)
        logger.info(f"[Model] Query shape: {Q.shape}, Key shape: {K.shape}, Value shape: {V.shape}")
        
        d_attn = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_attn)
        logger.info(f"[Model] Attention scores shape: {scores.shape}")
        
        mask_float = mask.float()
        mask_exp = mask_float.unsqueeze(1)
        scores = scores + (1 - mask_exp) * (-1e9)
        logger.info("[Model] Applied mask to attention scores.")
        
        attn_weights = F.softmax(scores, dim=-1)
        logger.info(f"[Model] Attention weights shape: {attn_weights.shape}")
        
        attn_vec = torch.bmm(attn_weights, V)
        logger.info(f"[Model] Aggregated attention vector shape: {attn_vec.shape}")
        
        winner_scores = torch.tanh(self.score_layer(attn_vec)).squeeze(-1)
        logger.info(f"[Model] Winner scores shape (pre-mask): {winner_scores.shape}")
        winner_scores = winner_scores.masked_fill(~mask, -1e9)
        logger.info("[Model] Applied mask to winner scores.")
        
        portfolio_weights = []
        for i in range(B):
            scores_i = winner_scores[i]
            valid_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
            logger.info(f"[Model] Batch {i}: valid asset indices: {valid_idx}")
            if valid_idx.numel() == 0:
                portfolio_weights.append(torch.zeros_like(scores_i))
                continue
            valid_scores = scores_i[valid_idx]
            G = self.G
            G_adj = min(G, valid_scores.size(0) // 2) if valid_scores.size(0) >= 2 else 1
            logger.info(f"[Model] Batch {i}: G_adj = {G_adj}")

            sorted_long = torch.argsort(valid_scores, descending=True)
            top_indices = valid_idx[sorted_long[:G_adj]]
            logger.info(f"[Model] Batch {i}: top_indices for long positions: {top_indices}")
            sorted_short = torch.argsort(valid_scores, descending=False)
            bottom_indices = valid_idx[sorted_short[:G_adj]]
            logger.info(f"[Model] Batch {i}: bottom_indices for short positions: {bottom_indices}")
            
            long_scores = scores_i[top_indices]
            long_weights = torch.softmax(long_scores, dim=0)
            logger.info(f"[Model] Batch {i}: long_weights: {long_weights}")
            short_scores = scores_i[bottom_indices]
            short_weights = torch.softmax(-short_scores, dim=0)
            logger.info(f"[Model] Batch {i}: short_weights: {short_weights}")
            
            b = torch.zeros_like(scores_i)
            b[top_indices] = long_weights
            b[bottom_indices] = -short_weights
            logger.info(f"[Model] Batch {i}: portfolio weights: {b}")
            portfolio_weights.append(b)
        portfolio_weights = torch.stack(portfolio_weights, dim=0)
        logger.info(f"[Model] Final portfolio_weights shape: {portfolio_weights.shape}")
        return portfolio_weights, winner_scores
 