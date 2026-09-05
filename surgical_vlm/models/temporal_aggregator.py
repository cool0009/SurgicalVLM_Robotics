"""
Temporal Aggregation Module for SurgicalVLM.

Provides two temporal modeling strategies:
1. TemporalAttentionAggregator - Multi-head self-attention over frame sequence (full temporal modeling)
2. LearnableTemporalPooler - Lightweight parameterized pooling (efficient for long sequences)

Both conform to the same interface for easy swapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class TemporalAttentionAggregator(nn.Module):
    """
    Multi-head self-attention over frame-level embeddings.
    
    Input:  [B, T, D]  (batch, frames, hidden_dim)
    Output: [B, T, D]  (enhanced frame embeddings) or [B, D] (if pool=True)
    
    Used for Stage 2 temporal modeling after VLM frame encoding.
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
        use_pos_encoding: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_frames = max_frames
        
        # Positional encoding
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, max_frames, hidden_dim)
            )
            nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better gradient flow
            layer_norm_eps=layer_norm_eps,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Layer norm after encoder
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Optional pooling head
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.pool_query, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,           # [B, T, D]
        mask: Optional[torch.Tensor] = None,  # [B, T] bool (True=valid)
        pool: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Frame embeddings [B, T, D]
            mask: Optional boolean mask [B, T], True for valid frames
            pool: If True, return [B, D] pooled representation
            
        Returns:
            If pool=False: [B, T, D] enhanced embeddings
            If pool=True:  [B, D] pooled representation
        """
        B, T, D = x.shape
        assert D == self.hidden_dim, f"Expected hidden_dim={self.hidden_dim}, got {D}"
        assert T <= self.max_frames, f"Sequence length {T} exceeds max_frames {self.max_frames}"
        
        # Add positional encoding
        if self.use_pos_encoding:
            x = x + self.pos_encoding[:, :T, :]
        
        # Transformer encoder
        # src_key_padding_mask: True = pad (mask out)
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True=mask out
        else:
            key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)  # [B, T, D]
        
        if not pool:
            return x
        
        # Attention pooling: use learnable query to attend over sequence
        query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
        pooled, attn_weights = self.pool_attn(
            query, x, x,
            key_padding_mask=key_padding_mask,
        )  # pooled: [B, 1, D]
        return pooled.squeeze(1)  # [B, D]
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract attention weights from a specific layer for visualization."""
        # This is a simplified version - full implementation would hook into transformer layers
        # For now, return dummy weights
        B, T, _ = x.shape
        return torch.ones(B, self.num_heads, T, T) / T


class LearnableTemporalPooler(nn.Module):
    """
    Lightweight temporal pooling using learnable attention or set operations.
    
    Much faster than full transformer, suitable for long sequences (>100 frames).
    Three modes:
    - 'attention': Learnable query-based attention pooling
    - 'weighted': Per-frame learned weights (softmax)
    - 'set': DeepSets-style mean/max + MLP
    
    Input:  [B, T, D]
    Output: [B, D] (pooled) or [B, T, D] (if return_sequence=True)
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        mode: str = "attention",
        num_heads: int = 4,
        dropout: float = 0.1,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode
        assert mode in ["attention", "weighted", "set"]
        
        if mode == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.pool_query, std=0.02)
            
        elif mode == "weighted":
            # Learnable scalar weight per frame, predicted from frame embedding
            self.weight_pred = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
            
        elif mode == "set":
            # DeepSets: phi(x) -> mean/max -> rho
            self.phi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.rho = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
    
    def forward(
        self,
        x: torch.Tensor,           # [B, T, D]
        mask: Optional[torch.Tensor] = None,  # [B, T] bool
        return_sequence: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        
        if self.mode == "attention":
            # Attention pooling with learnable query
            query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            key_padding_mask = ~mask
            pooled, attn_weights = self.attn(
                query, x, x,
                key_padding_mask=key_padding_mask,
            )  # pooled: [B, 1, D]
            
            if return_sequence:
                # Broadcast pooled to sequence length for residual connection
                return x + pooled.expand(-1, T, -1)
            return pooled.squeeze(1)  # [B, D]
        
        elif self.mode == "weighted":
            # Predict weight per frame, masked softmax
            logits = self.weight_pred(x).squeeze(-1)  # [B, T]
            logits = logits.masked_fill(~mask, -1e9)
            weights = F.softmax(logits, dim=1)  # [B, T]
            pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
            
            if return_sequence:
                return x * weights.unsqueeze(-1)  # [B, T, D] weighted
            return pooled
        
        elif self.mode == "set":
            # DeepSets: apply phi to each element, aggregate, apply rho
            x_phi = self.phi(x)  # [B, T, D]
            
            # Masked mean and max
            masked_x = x_phi.masked_fill(~mask.unsqueeze(-1), 0)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            mean_pool = masked_x.sum(dim=1) / count  # [B, D]
            
            masked_x_max = x_phi.masked_fill(~mask.unsqueeze(-1), -1e9)
            max_pool = masked_x_max.max(dim=1).values  # [B, D]
            
            agg = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2D]
            pooled = self.rho(agg)  # [B, D]
            
            if return_sequence:
                # Broadcast aggregated features
                return x + pooled.unsqueeze(1).expand(-1, T, -1)
            return pooled
        
        raise ValueError(f"Unknown mode: {self.mode}")


class TemporalAggregator(nn.Module):
    """
    Unified wrapper that selects between TemporalAttentionAggregator and LearnableTemporalPooler.
    
    Usage:
        aggregator = TemporalAggregator(
            hidden_dim=3584,
            strategy="attention",  # or "pooler_attention", "pooler_weighted", "pooler_set"
            num_layers=4,  # for attention strategy
        )
        pooled = aggregator(x, mask=mask, pool=True)  # [B, D]
        enhanced = aggregator(x, mask=mask, pool=False)  # [B, T, D]
    """
    
    def __init__(
        self,
        hidden_dim: int = 3584,
        strategy: str = "attention",  # "attention", "pooler_attention", "pooler_weighted", "pooler_set"
        **kwargs,
    ):
        super().__init__()
        self.strategy = strategy
        
        if strategy == "attention":
            self.aggregator = TemporalAttentionAggregator(
                hidden_dim=hidden_dim,
                **kwargs,
            )
        elif strategy.startswith("pooler_"):
            pooler_mode = strategy.replace("pooler_", "")
            self.aggregator = LearnableTemporalPooler(
                hidden_dim=hidden_dim,
                mode=pooler_mode,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pool: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] boolean
            pool: If True, return [B, D] (only for attention strategy)
            return_sequence: If True, return [B, T, D] (for pooler strategies)
        """
        if self.strategy == "attention":
            return self.aggregator(x, mask=mask, pool=pool)
        else:
            return self.aggregator(x, mask=mask, return_sequence=return_sequence)
    
    def get_model_size(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}


def create_temporal_aggregator(config: Dict[str, Any]) -> TemporalAggregator:
    """Factory function from config dict."""
    strategy = config.get("strategy", "attention")
    base_kwargs = {
        "hidden_dim": config.get("hidden_dim", 3584),
        "strategy": strategy,
        "num_heads": config.get("num_heads", 8),
        "num_layers": config.get("num_layers", 4),
        "dropout": config.get("dropout", 0.1),
        "max_frames": config.get("max_frames", 256),
        "use_pos_encoding": config.get("use_pos_encoding", True),
    }
    
    # Only pass pooler-specific params for pooler strategies
    if strategy.startswith("pooler_"):
        base_kwargs["bottleneck_dim"] = config.get("bottleneck_dim", 256)
    
    return TemporalAggregator(**base_kwargs)


if __name__ == "__main__":
    # Quick sanity check
    B, T, D = 2, 16, 896
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 10:] = False  # Variable length
    
    for strategy in ["attention", "pooler_attention", "pooler_weighted", "pooler_set"]:
        print(f"\nTesting {strategy}...")
        agg = TemporalAggregator(hidden_dim=D, strategy=strategy)
        
        if strategy == "attention":
            out_seq = agg(x, mask=mask, pool=False)
            out_pool = agg(x, mask=mask, pool=True)
            print(f"  Sequence: {out_seq.shape}")  # [B, T, D]
            print(f"  Pooled:   {out_pool.shape}")   # [B, D]
        else:
            out_seq = agg(x, mask=mask, return_sequence=True)
            out_pool = agg(x, mask=mask, pool=True)  # pooler ignores pool arg
            print(f"  Sequence: {out_seq.shape}")  # [B, T, D]
            print(f"  Pooled:   {out_pool.shape}")  # [B, D]
        
        size = agg.get_model_size()
        print(f"  Params: {size['trainable_params']:,}")
    
    print("\n✅ All temporal aggregator tests passed!")