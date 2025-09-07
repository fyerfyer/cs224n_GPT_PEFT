"""
LoRA-enabled Causal Self-Attention module for parameter-efficient fine-tuning.

This module provides a LoRA version of the CausalSelfAttention that can be used
as a drop-in replacement for the original attention mechanism.
"""

import math
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from modules.lora import LoRALayer


class LoRACausalSelfAttention(nn.Module):
    """
    LoRA-enabled Causal Self-Attention module.
    
    This is identical to the original CausalSelfAttention but uses LoRA layers
    for the query, key, value, and output projections to enable parameter-efficient
    fine-tuning.
    """
    
    def __init__(self, config, lora_config=None):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # LoRA configuration
        lora_config = lora_config or {}
        self.lora_rank = lora_config.get('rank', 4)
        self.lora_alpha = lora_config.get('alpha', 16.0)
        self.lora_dropout = lora_config.get('dropout', 0.0)

        # Initialize LoRA layers for key, value, query projections
        self.query = LoRALayer(
            config.hidden_size, 
            self.all_head_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        self.key = LoRALayer(
            config.hidden_size, 
            self.all_head_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        self.value = LoRALayer(
            config.hidden_size, 
            self.all_head_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout
        )
        
        # Attention dropout (same as original)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    @classmethod
    def from_pretrained_attention(cls, original_attention, config, lora_config=None):
        """
        Create a LoRA attention module from a pretrained attention module.
        
        Args:
            original_attention: The original CausalSelfAttention module
            config: Model configuration
            lora_config: LoRA configuration dict
            
        Returns:
            LoRACausalSelfAttention with loaded pretrained weights
        """
        lora_attention = cls(config, lora_config)
        
        # Load pretrained weights into LoRA layers
        lora_attention.query.load_pretrained_weights(
            original_attention.query.weight.data,
            original_attention.query.bias.data if original_attention.query.bias is not None else None
        )
        lora_attention.key.load_pretrained_weights(
            original_attention.key.weight.data,
            original_attention.key.bias.data if original_attention.key.bias is not None else None
        )
        lora_attention.value.load_pretrained_weights(
            original_attention.value.weight.data,
            original_attention.value.bias.data if original_attention.value.bias is not None else None
        )
        
        return lora_attention

    def transform(self, x, lora_layer):
        """
        Transform input using LoRA layer and reshape for multi-head attention.
        
        Args:
            x: Input tensor
            lora_layer: LoRALayer for projection
            
        Returns:
            Transformed tensor ready for attention computation
        """
        # Project using LoRA layer
        proj = lora_layer(x)
        # Reshape for multi-head attention
        proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
        proj = rearrange(proj, 'b t h d -> b h t d')
        return proj

    def attention(self, key, query, value, attention_mask):
        """
        Compute scaled dot-product attention with causal masking.
        
        Args:
            key: Key tensor [B, nh, T, d_k]
            query: Query tensor [B, nh, T, d_k]
            value: Value tensor [B, nh, T, d_k]
            attention_mask: Attention mask [B, 1, 1, T]
            
        Returns:
            Attention output [B, nh, T, d_k]
        """
        # Calculate attention scores
        # (B, nh, T, d_k) x (B, nh, d_k, T) -> (B, nh, T, T)
        att = torch.matmul(query, key.transpose(-2, -1))
        att = att / math.sqrt(query.size(-1))

        # Apply the causal mask
        seq_len = query.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        # Apply the attention mask if provided
        if attention_mask is not None:
            att = att + attention_mask

        # Normalize the scores to get attention weights
        att = F.softmax(att, dim=-1)

        # Apply dropout
        att = self.dropout(att)

        # Multiply weights by values
        # (B, nh, T, T) x (B, nh, T, d_k) -> (B, nh, T, d_k)
        att = torch.matmul(att, value)

        return att

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass of LoRA-enabled causal self-attention.
        
        Args:
            hidden_states: Input hidden states [bs, seq_len, hidden_size]
            attention_mask: Attention mask [bs, 1, 1, seq_len]
            
        Returns:
            Attention output [bs, seq_len, hidden_size]
        """
        # Generate key, value, query using LoRA layers
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        
        # Calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        
        # Reshape back to original format
        attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')
        
        return attn_value