"""
ReFT-enabled GPT-2 Layer implementation for parameter-efficient fine-tuning.

This module provides a ReFT version of the GPT2Layer that applies representation
interventions during the forward pass rather than modifying weights.
"""

import torch
from torch import nn
import torch.nn.functional as F

from modules.attention import CausalSelfAttention
from modules.reft import ReFTIntervention, ReFTConfig


class ReFTGPT2Layer(nn.Module):
    """
    ReFT-enabled GPT-2 Layer.
    
    This layer is identical to the original GPT2Layer but applies ReFT interventions
    to hidden states during the forward pass to enable parameter-efficient fine-tuning.
    All original weights remain frozen - only the ReFT intervention parameters are trainable.
    """
    
    def __init__(self, config, reft_config=None):
        super().__init__()
        
        # ReFT configuration
        if isinstance(reft_config, dict):
            reft_config = ReFTConfig.from_dict(reft_config)
        elif reft_config is None:
            reft_config = ReFTConfig()
        
        self.reft_config = reft_config
        
        # Original GPT-2 components (these will be frozen)
        self.self_attention = CausalSelfAttention(config)
        
        # Attention output projection (frozen)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer normalization and dropout (frozen)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Feed-forward network (frozen)
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Output layer normalization and dropout (frozen)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # ReFT intervention modules (only these are trainable)
        self.reft_interventions = nn.ModuleDict()
        
        # Apply interventions based on configuration
        if 'attention' in reft_config.intervention_locations:
            self.reft_interventions['attention'] = ReFTIntervention(
                hidden_size=config.hidden_size,
                rank=reft_config.rank,
                alpha=reft_config.alpha,
                dropout=reft_config.dropout,
                activation=reft_config.activation
            )
            
        if 'ffn' in reft_config.intervention_locations:
            self.reft_interventions['ffn'] = ReFTIntervention(
                hidden_size=config.hidden_size,
                rank=reft_config.rank,
                alpha=reft_config.alpha,
                dropout=reft_config.dropout,
                activation=reft_config.activation
            )
        
        # Freeze all original parameters
        self.freeze_original_parameters()

    def freeze_original_parameters(self):
        """Freeze all original GPT-2 parameters, keeping only ReFT parameters trainable."""
        for name, param in self.named_parameters():
            if 'reft_' not in name:
                param.requires_grad = False

    def unfreeze_original_parameters(self):
        """Unfreeze original parameters (if needed for debugging)."""
        for name, param in self.named_parameters():
            if 'reft_' not in name:
                param.requires_grad = True

    @classmethod
    def from_pretrained_layer(cls, original_layer, config, reft_config=None):
        """
        Create a ReFT layer from a pretrained GPT2Layer.
        
        Args:
            original_layer: The original GPT2Layer
            config: Model configuration
            reft_config: ReFT configuration
            
        Returns:
            ReFTGPT2Layer with loaded pretrained weights
        """
        reft_layer = cls(config, reft_config)
        
        # Load all original weights
        reft_layer.self_attention.load_state_dict(original_layer.self_attention.state_dict())
        reft_layer.attention_dense.load_state_dict(original_layer.attention_dense.state_dict())
        reft_layer.interm_dense.load_state_dict(original_layer.interm_dense.state_dict())
        reft_layer.out_dense.load_state_dict(original_layer.out_dense.state_dict())
        
        # Load layer norm parameters
        reft_layer.attention_layer_norm.load_state_dict(original_layer.attention_layer_norm.state_dict())
        reft_layer.out_layer_norm.load_state_dict(original_layer.out_layer_norm.state_dict())
        
        # Re-freeze parameters after loading
        reft_layer.freeze_original_parameters()
        
        return reft_layer

    def add(self, input_tensor, output_tensor, dense_layer, dropout):
        """
        Helper method for residual connection with dropout.
        
        Args:
            input_tensor: Input tensor for residual connection
            output_tensor: Output tensor to transform
            dense_layer: Dense layer for transformation
            dropout: Dropout module
            
        Returns:
            Result of residual connection
        """
        output = dropout(dense_layer(output_tensor))
        return input_tensor + output

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass of ReFT-enabled GPT-2 layer.
        
        Args:
            hidden_states: Input hidden states [bs, seq_len, hidden_size]
            attention_mask: Attention mask [bs, 1, 1, seq_len]
            
        Returns:
            Output hidden states [bs, seq_len, hidden_size]
        """
        # Self-attention with pre-layer norm (same as original)
        ln_output = self.attention_layer_norm(hidden_states)
        att_output = self.self_attention(ln_output, attention_mask)
        hidden_states = self.add(hidden_states, att_output, self.attention_dense, self.attention_dropout)
        
        # Apply ReFT intervention after attention if configured
        if 'attention' in self.reft_interventions:
            hidden_states = self.reft_interventions['attention'](hidden_states)
        
        # Feed-forward with pre-layer norm (same as original)
        ln_output = self.out_layer_norm(hidden_states)
        interm_output = self.interm_af(self.interm_dense(ln_output))
        hidden_states = self.add(hidden_states, interm_output, self.out_dense, self.out_dropout)
        
        # Apply ReFT intervention after FFN if configured
        if 'ffn' in self.reft_interventions:
            hidden_states = self.reft_interventions['ffn'](hidden_states)
        
        return hidden_states

    def get_reft_parameters(self):
        """
        Get all ReFT parameters for optimization.
        
        Returns:
            Generator of ReFT parameters
        """
        for name, param in self.named_parameters():
            if 'reft_' in name and param.requires_grad:
                yield param

    def get_reft_named_parameters(self):
        """
        Get all named ReFT parameters.
        
        Returns:
            Generator of (name, parameter) tuples for ReFT parameters
        """
        for name, param in self.named_parameters():
            if 'reft_' in name and param.requires_grad:
                yield name, param

    def print_intervention_info(self):
        """Print information about ReFT interventions in this layer."""
        print(f"ReFT Interventions: {list(self.reft_interventions.keys())}")
        print(f"ReFT Config: rank={self.reft_config.rank}, alpha={self.reft_config.alpha}, "
              f"dropout={self.reft_config.dropout}, activation={self.reft_config.activation}")
        
        # Count ReFT parameters
        reft_params = sum(p.numel() for p in self.get_reft_parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ReFT params: {reft_params:,} || Total params: {total_params:,} || "
              f"ReFT%: {100 * reft_params / total_params:.2f}%")