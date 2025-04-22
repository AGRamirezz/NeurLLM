"""
Models module for NeurLLM.

This module defines transformer-based models for processing neurophysiological data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class NeuroEncoder(nn.Module):
    """Base encoder class for different neurophysiological modalities."""
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Dimension of input features
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, embed_dim))  # Max 5000 time steps
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Encoded representation of shape (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        return x


class MultimodalFusion(nn.Module):
    """Module for fusing multiple modalities."""
    
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int):
        """
        Initialize the fusion module.
        
        Args:
            modality_dims: Dictionary mapping modality names to their dimensions
            fusion_dim: Dimension of the fused representation
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        
        # Projections for each modality
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, fusion_dim)
            for modality, dim in modality_dims.items()
        })
        
        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, modality_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            modality_outputs: Dictionary mapping modality names to their encoded representations
            
        Returns:
            Fused representation
        """
        # Project each modality to fusion dimension
        projected = {}
        for modality, tensor in modality_outputs.items():
            projected[modality] = self.projections[modality](tensor)
        
        # Stack all projections
        stacked = torch.stack(list(projected.values()), dim=1)  # (batch_size, num_modalities, seq_len, fusion_dim)
        batch_size, num_modalities, seq_len, fusion_dim = stacked.shape
        
        # Reshape for cross-attention
        stacked = stacked.view(batch_size, num_modalities * seq_len, fusion_dim)
        
        # Apply cross-attention for fusion
        fused, _ = self.cross_attention(stacked, stacked, stacked)
        fused = self.norm(fused + stacked)  # Residual connection
        
        return fused


class NeurLLM(nn.Module):
    """Main model class for NeurLLM."""
    
    def __init__(
        self,
        modality_configs: Dict[str, Dict],
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        vocab_size: int = 50257,  # GPT-2 vocabulary size
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize the model.
        
        Args:
            modality_configs: Configuration for each modality encoder
            hidden_dim: Hidden dimension of the model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            vocab_size: Size of the vocabulary for language modeling
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create encoders for each modality
        self.encoders = nn.ModuleDict({
            modality: NeuroEncoder(
                input_dim=config.get("input_dim", 64),
                embed_dim=config.get("embed_dim", hidden_dim),
                num_heads=config.get("num_heads", num_heads),
                num_layers=config.get("num_layers", 4),
                dropout=dropout
            )
            for modality, config in modality_configs.items()
        })
        
        # Multimodal fusion
        modality_dims = {
            modality: config.get("embed_dim", hidden_dim)
            for modality, config in modality_configs.items()
        }
        self.fusion = MultimodalFusion(modality_dims, hidden_dim)
        
        # Transformer decoder for language modeling
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights of the model."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        inputs: Dict[str, torch.Tensor],
        text_input: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
            text_input: Optional text input for decoder
            text_mask: Optional mask for text input
            
        Returns:
            Output logits
        """
        # Encode each modality
        encoded = {}
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                encoded[modality] = encoder(inputs[modality])
        
        # Fuse modalities
        fused = self.fusion(encoded)
        
        # Decode with transformer decoder if text input is provided
        if text_input is not None:
            output = self.decoder(text_input, fused, tgt_key_padding_mask=text_mask)
        else:
            output = fused
        
        # Project to vocabulary
        logits = self.output_layer(output)
        
        return logits 