"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.

This module provides LoRA layers that can replace standard Linear layers to enable
parameter-efficient fine-tuning by learning low-rank decomposition matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer implementation.
    
    This layer replaces a standard nn.Linear layer and adds trainable low-rank matrices
    A and B such that the adapted weight is: W = W_0 + B @ A, where W_0 is frozen.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of the low-rank decomposition (r in the paper)
        alpha: Scaling parameter (α in the paper)
        dropout: Dropout probability for LoRA layers
        bias: Whether to include bias (copied from original layer)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        
        # Original linear layer (will be frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA matrices
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_dropout = nn.Dropout(dropout)
            
            # Initialize LoRA matrices
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Original linear transformation (frozen)
        result = self.linear(x)
        
        # LoRA adaptation
        if self.rank > 0:
            # Compute low-rank adaptation: x @ A^T @ B^T
            lora_result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            # Apply scaling and add to original result
            result = result + lora_result * self.scaling
            
        return result
    
    def load_pretrained_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Load pretrained weights into the frozen linear layer.
        
        Args:
            weight: Pretrained weight matrix
            bias: Pretrained bias vector (optional)
        """
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            if bias is not None and self.linear.bias is not None:
                self.linear.bias.copy_(bias)


def convert_linear_to_lora(
    linear_layer: nn.Linear,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0
) -> LoRALayer:
    """
    Convert a standard nn.Linear layer to a LoRALayer while preserving weights.
    
    Args:
        linear_layer: The original nn.Linear layer
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
        
    Returns:
        LoRALayer with the same weights as the original layer
    """
    lora_layer = LoRALayer(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=linear_layer.bias is not None
    )
    
    # Copy the weights from the original layer
    lora_layer.load_pretrained_weights(
        linear_layer.weight.data,
        linear_layer.bias.data if linear_layer.bias is not None else None
    )
    
    return lora_layer