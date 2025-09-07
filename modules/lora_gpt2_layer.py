"""
LoRA-enabled GPT-2 Layer implementation for parameter-efficient fine-tuning.

This module provides a LoRA version of the GPT2Layer that replaces linear layers
with LoRA layers to enable parameter-efficient fine-tuning.
"""

from torch import nn
import torch.nn.functional as F

from modules.lora_attention import LoRACausalSelfAttention
from modules.lora import LoRALayer


class LoRAGPT2Layer(nn.Module):
    """
    LoRA-enabled GPT-2 Layer.
    
    This layer is identical to the original GPT2Layer but uses LoRA layers
    for the attention projections and feed-forward network to enable 
    parameter-efficient fine-tuning.
    """
    
    def __init__(self, config, lora_config=None):
        super().__init__()
        
        # LoRA configuration
        lora_config = lora_config or {}
        self.lora_rank = lora_config.get('rank', 4)
        self.lora_alpha = lora_config.get('alpha', 16.0)
        self.lora_dropout = lora_config.get('dropout', 0.0)
        
        # LoRA-enabled multi-head attention
        self.self_attention = LoRACausalSelfAttention(config, lora_config)
        
        # Attention output projection with LoRA
        self.attention_dense = LoRALayer(
            config.hidden_size, 
            config.hidden_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        
        # Layer normalization and dropout (same as original)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Feed-forward network with LoRA
        self.interm_dense = LoRALayer(
            config.hidden_size, 
            config.intermediate_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        self.interm_af = F.gelu
        
        self.out_dense = LoRALayer(
            config.intermediate_size, 
            config.hidden_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        
        # Output layer normalization and dropout (same as original)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    @classmethod
    def from_pretrained_layer(cls, original_layer, config, lora_config=None):
        """
        Create a LoRA layer from a pretrained GPT2Layer.
        
        Args:
            original_layer: The original GPT2Layer
            config: Model configuration
            lora_config: LoRA configuration dict
            
        Returns:
            LoRAGPT2Layer with loaded pretrained weights
        """
        lora_layer = cls(config, lora_config)
        
        # Load attention weights
        lora_layer.self_attention = LoRACausalSelfAttention.from_pretrained_attention(
            original_layer.self_attention, config, lora_config
        )
        
        # Load attention dense layer weights
        lora_layer.attention_dense.load_pretrained_weights(
            original_layer.attention_dense.weight.data,
            original_layer.attention_dense.bias.data if original_layer.attention_dense.bias is not None else None
        )
        
        # Load feed-forward weights
        lora_layer.interm_dense.load_pretrained_weights(
            original_layer.interm_dense.weight.data,
            original_layer.interm_dense.bias.data if original_layer.interm_dense.bias is not None else None
        )
        lora_layer.out_dense.load_pretrained_weights(
            original_layer.out_dense.weight.data,
            original_layer.out_dense.bias.data if original_layer.out_dense.bias is not None else None
        )
        
        # Copy layer norm parameters (these will remain trainable)
        lora_layer.attention_layer_norm.load_state_dict(original_layer.attention_layer_norm.state_dict())
        lora_layer.out_layer_norm.load_state_dict(original_layer.out_layer_norm.state_dict())
        
        return lora_layer

    def add(self, input_tensor, output_tensor, lora_layer, dropout):
        """
        Helper method for residual connection with dropout.
        
        Args:
            input_tensor: Input tensor for residual connection
            output_tensor: Output tensor to transform
            lora_layer: LoRA layer for transformation
            dropout: Dropout module
            
        Returns:
            Result of residual connection
        """
        output = dropout(lora_layer(output_tensor))
        return input_tensor + output

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass of LoRA-enabled GPT-2 layer.
        
        Args:
            hidden_states: Input hidden states [bs, seq_len, hidden_size]
            attention_mask: Attention mask [bs, 1, 1, seq_len]
            
        Returns:
            Output hidden states [bs, seq_len, hidden_size]
        """
        # Self-attention with pre-layer norm
        ln_output = self.attention_layer_norm(hidden_states)
        att_output = self.self_attention(ln_output, attention_mask)
        hidden_states = self.add(hidden_states, att_output, self.attention_dense, self.attention_dropout)
        
        # Feed-forward with pre-layer norm
        ln_output = self.out_layer_norm(hidden_states)
        interm_output = self.interm_af(self.interm_dense(ln_output))
        hidden_states = self.add(hidden_states, interm_output, self.out_dense, self.out_dropout)
        
        return hidden_states