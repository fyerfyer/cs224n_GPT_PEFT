"""
ReFT-enabled GPT-2 Model for parameter-efficient fine-tuning.

This module provides a ReFT version of the GPT2Model that enables parameter-efficient
fine-tuning by using ReFT intervention layers that modify representations during
the forward pass rather than modifying weights.
"""

import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from models.gpt2 import GPT2Model
from modules.reft_gpt2_layer import ReFTGPT2Layer
from modules.reft import ReFTConfig
from utils import get_extended_attention_mask


class ReFTGPT2Model(GPTPreTrainedModel):
    """
    ReFT-enabled GPT-2 model for parameter-efficient fine-tuning.
    
    This model is identical to GPT2Model but uses ReFT intervention layers to enable
    parameter-efficient fine-tuning while keeping the original weights frozen.
    Only the ReFT intervention parameters are trainable.
    """

    def __init__(self, config, reft_config=None):
        super().__init__(config)
        self.config = config
        
        # ReFT configuration
        if isinstance(reft_config, dict):
            self.reft_config = ReFTConfig.from_dict(reft_config)
        elif reft_config is None:
            self.reft_config = ReFTConfig()
        else:
            self.reft_config = reft_config

        # Embedding layers (these remain the same as original and will be frozen)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # ReFT-enabled GPT-2 layers
        self.gpt_layers = nn.ModuleList([
            ReFTGPT2Layer(config, self.reft_config) 
            if self.reft_config.intervention_layers is None or i in self.reft_config.intervention_layers
            else ReFTGPT2Layer(config, None)  # No intervention for this layer
            for i in range(config.num_hidden_layers)
        ])

        # [CLS] token transformations (keep original and freeze)
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        # Final layer norm (keep original and freeze)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()
        
        # Freeze all non-ReFT parameters by default
        self.freeze_base_parameters()

    def freeze_base_parameters(self):
        """Freeze all base model parameters (embeddings, layers, etc.), keeping only ReFT parameters trainable."""
        for name, param in self.named_parameters():
            if 'reft_' not in name:
                param.requires_grad = False

    def unfreeze_embeddings(self):
        """Unfreeze the embedding layers if needed."""
        for param in self.word_embedding.parameters():
            param.requires_grad = True
        for param in self.pos_embedding.parameters():
            param.requires_grad = True

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

    def print_trainable_parameters(self):
        """Print statistics about trainable vs total parameters."""
        trainable_params = 0
        reft_params = 0
        all_param = 0
        
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'reft_' in name:
                    reft_params += param.numel()
        
        print(f"ReFT params: {reft_params:,} || "
              f"Trainable params: {trainable_params:,} || "
              f"All params: {all_param:,} || "
              f"Trainable%: {100 * trainable_params / all_param:.2f}%")

    @classmethod
    def from_pretrained_gpt2(cls, original_model: GPT2Model, reft_config=None):
        """
        Create a ReFT model from a pretrained GPT2Model.
        
        Args:
            original_model: The pretrained GPT2Model
            reft_config: ReFT configuration dict or ReFTConfig object
            
        Returns:
            ReFTGPT2Model with loaded pretrained weights
        """
        config = original_model.config
        reft_model = cls(config, reft_config)
        
        # Copy embedding weights
        reft_model.word_embedding.load_state_dict(original_model.word_embedding.state_dict())
        reft_model.pos_embedding.load_state_dict(original_model.pos_embedding.state_dict())
        
        # Copy pooler weights
        reft_model.pooler_dense.load_state_dict(original_model.pooler_dense.state_dict())
        
        # Copy final layer norm weights
        reft_model.final_layer_norm.load_state_dict(original_model.final_layer_norm.state_dict())
        
        # Convert each layer to ReFT
        for i, original_layer in enumerate(original_model.gpt_layers):
            # Determine if this layer should have ReFT interventions
            layer_reft_config = reft_model.reft_config if (
                reft_model.reft_config.intervention_layers is None or 
                i in reft_model.reft_config.intervention_layers
            ) else None
            
            reft_model.gpt_layers[i] = ReFTGPT2Layer.from_pretrained_layer(
                original_layer, config, layer_reft_config
            )
        
        # Re-freeze base parameters after loading
        reft_model.freeze_base_parameters()
        
        return reft_model

    @classmethod
    def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12, reft_config=None):
        """
        Create a ReFT model from a pretrained OpenAI GPT-2 model.
        
        Args:
            model: Model name or path
            d: Hidden size
            l: Number of layers
            num_heads: Number of attention heads
            reft_config: ReFT configuration dict or ReFTConfig object
            
        Returns:
            ReFTGPT2Model with loaded pretrained weights
        """
        # First create the original GPT2Model
        original_model = GPT2Model.from_pretrained(model, d, l, num_heads)
        
        # Then convert it to ReFT
        return cls.from_pretrained_gpt2(original_model, reft_config)

    def embed(self, input_ids):
        """Embedding layer (same as original)."""
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)

        embeddings = inputs_embeds + pos_embeds
        embeddings = self.embed_dropout(embeddings)
        return embeddings

    def encode(self, hidden_states, attention_mask):
        """
        Encode hidden states through ReFT-enabled layers.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Encoded hidden states
        """
        extended_attention_mask = get_extended_attention_mask(attention_mask, self.dtype)

        for layer_module in self.gpt_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of ReFT-enabled GPT-2.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dict with 'last_hidden_state' and 'last_token' keys
        """
        # Get embeddings
        embedding_output = self.embed(input_ids=input_ids)

        # Encode through ReFT layers
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        # Get the hidden state of the final token
        # attention_mask can be a float tensor; make sure the computed indices are integer type
        # also clamp to 0 to avoid negative indices when a sequence might be entirely padding
        last_non_pad_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()
        last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

        return {'last_hidden_state': sequence_output, 'last_token': last_token}

    def hidden_state_to_token(self, hidden_state):
        """
        Convert hidden state to token logits using weight tying.
        
        Args:
            hidden_state: Hidden state tensor
            
        Returns:
            Token logits
        """
        return torch.matmul(hidden_state, self.word_embedding.weight.T)

    def print_reft_config(self):
        """Print the ReFT configuration details."""
        print("ReFT Configuration:")
        print(f"  Rank: {self.reft_config.rank}")
        print(f"  Alpha: {self.reft_config.alpha}")
        print(f"  Dropout: {self.reft_config.dropout}")
        print(f"  Activation: {self.reft_config.activation}")
        print(f"  Intervention locations: {self.reft_config.intervention_locations}")
        print(f"  Intervention layers: {self.reft_config.intervention_layers or 'All layers'}")
        
        # Count intervention points
        intervention_count = 0
        for i, layer in enumerate(self.gpt_layers):
            if hasattr(layer, 'reft_interventions'):
                intervention_count += len(layer.reft_interventions)
        
        print(f"  Total intervention points: {intervention_count}")