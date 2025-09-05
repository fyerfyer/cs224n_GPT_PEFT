"""
ReFT (Representation Fine-Tuning) implementation for parameter-efficient fine-tuning.

This module provides ReFT intervention layers that modify hidden states during
the forward pass rather than modifying model weights like LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ReFTIntervention(nn.Module):
    """
    ReFT (Representation Fine-Tuning) intervention module.
    
    Unlike LoRA which modifies weights, ReFT intervenes on representations (hidden states)
    during the forward pass by learning a small, low-rank function that modifies activations.
    
    The intervention is formulated as:
    h_modified = h + W_up(activation(W_down(h)))
    
    Args:
        hidden_size: Size of hidden states to intervene on
        rank: Rank of the low-rank decomposition (r in ReFT)
        alpha: Scaling parameter (α in ReFT)
        dropout: Dropout probability for ReFT layers
        activation: Activation function for the intervention ('relu', 'gelu', 'tanh')
    """
    
    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        
        # ReFT intervention matrices (low-rank decomposition)
        if rank > 0:
            # Down projection: hidden_size -> rank
            self.reft_down = nn.Parameter(torch.zeros(rank, hidden_size))
            # Up projection: rank -> hidden_size  
            self.reft_up = nn.Parameter(torch.zeros(hidden_size, rank))
            self.reft_dropout = nn.Dropout(dropout)
            
            # Initialize ReFT matrices
            nn.init.kaiming_uniform_(self.reft_down, a=math.sqrt(5))
            nn.init.zeros_(self.reft_up)
            
            # Activation function for the intervention
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply ReFT intervention to hidden states.
        
        Args:
            hidden_states: Input hidden states [..., hidden_size]
            
        Returns:
            Modified hidden states [..., hidden_size]
        """
        if self.rank <= 0:
            return hidden_states
            
        # Original hidden states (unmodified)
        original_states = hidden_states
        
        # Apply ReFT intervention: h + W_up(activation(W_down(h)))
        # Down projection: [..., hidden_size] -> [..., rank]
        down_proj = torch.matmul(hidden_states, self.reft_down.T)
        
        # Apply activation
        activated = self.activation(down_proj)
        
        # Apply dropout
        activated = self.reft_dropout(activated)
        
        # Up projection: [..., rank] -> [..., hidden_size]
        up_proj = torch.matmul(activated, self.reft_up.T)
        
        # Apply scaling and add to original states (residual connection)
        intervention = up_proj * self.scaling
        modified_states = original_states + intervention
        
        return modified_states


class LoReFTIntervention(ReFTIntervention):
    """
    LoReFT (Low-Rank Representation Fine-Tuning) - A specific variant of ReFT.
    
    This is essentially the same as ReFTIntervention but with a specific naming
    convention to match the LoReFT paper terminology.
    """
    
    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        super().__init__(hidden_size, rank, alpha, dropout, activation)


def create_reft_intervention(
    hidden_size: int,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0,
    activation: str = 'relu'
) -> ReFTIntervention:
    """
    Factory function to create a ReFT intervention module.
    
    Args:
        hidden_size: Size of hidden states
        rank: ReFT rank
        alpha: ReFT alpha parameter
        dropout: ReFT dropout rate
        activation: Activation function name
        
    Returns:
        ReFTIntervention module
    """
    return ReFTIntervention(
        hidden_size=hidden_size,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        activation=activation
    )


class ReFTConfig:
    """Configuration class for ReFT interventions."""
    
    def __init__(
        self,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        intervention_layers: Optional[list] = None,
        intervention_locations: Optional[list] = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.activation = activation
        # Which transformer layers to apply interventions to (None = all layers)
        self.intervention_layers = intervention_layers
        # Where in each layer to apply interventions ('attention', 'ffn', 'both')
        self.intervention_locations = intervention_locations or ['attention']
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'activation': self.activation,
            'intervention_layers': self.intervention_layers,
            'intervention_locations': self.intervention_locations
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)