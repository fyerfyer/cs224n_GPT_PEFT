'''
LoRA-enabled Sonnet generation.

Running:
  `python lora_sonnet_generation.py --use_gpu`

trains your LoRASonnetGPT model using LoRA fine-tuning and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.lora_gpt2 import LoRAGPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class LoRASonnetGPT(nn.Module):
  """LoRA-enabled GPT-2 Model designed for sonnet generation."""

  def __init__(self, args):
    super().__init__()
    
    # LoRA configuration
    lora_config = {
      'rank': getattr(args, 'lora_rank', 4),
      'alpha': getattr(args, 'lora_alpha', 16.0),
      'dropout': getattr(args, 'lora_dropout', 0.0)
    }
    
    # Create LoRA-enabled GPT-2 model
    self.gpt = LoRAGPT2Model.from_pretrained(
      model=args.model_size, 
      d=args.d, 
      l=args.l, 
      num_heads=args.num_heads,
      lora_config=lora_config
    )
    
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Print LoRA parameter statistics
    self.gpt.print_trainable_parameters()

  def forward(self, input_ids, attention_mask):
    """
    Forward pass for sonnet generation training.
    
    This returns logits for each token in our sequence to enable language modeling loss.
    Unlike paraphrase detection, we want to predict the next token for each position
    in the sequence to learn the natural language distribution of sonnets.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Token logits for all positions [batch_size, seq_len, vocab_size]
    """
    # Get GPT-2 output - this gives us hidden states for all positions
    outputs = self.gpt(input_ids, attention_mask)
    sequence_output = outputs['last_hidden_state']  # [batch_size, seq_len, hidden_size]
    
    # Convert hidden states to token logits for all positions using weight tying
    logits = self.gpt.hidden_state_to_token(sequence_output)  # [batch_size, seq_len, vocab_size]
    
    return logits

  def get_lora_parameters(self):
    """Get only LoRA parameters for optimization."""
    return self.gpt.get_lora_parameters()

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train LoRA-enabled GPT-2 for sonnet generation."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Create the data and its corresponding datasets and dataloader.
  full_sonnet_dataset = SonnetsDataset(args.sonnet_path)
  
  # Create 90/10 train/validation split
  train_size = int(0.9 * len(full_sonnet_dataset))
  val_size = len(full_sonnet_dataset) - train_size
  train_dataset, val_dataset = random_split(full_sonnet_dataset, [train_size, val_size])
  
  print(f"Training on {train_size} sonnets, validating on {val_size} sonnets")
  
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                               collate_fn=full_sonnet_dataset.collate_fn)
  val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                             collate_fn=full_sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = LoRASonnetGPT(args)
  model = model.to(device)

  # Create optimizer with only LoRA parameters
  lora_parameters = list(model.get_lora_parameters())
  if not lora_parameters:
    raise ValueError("No LoRA parameters found! Check LoRA implementation.")
  
  print(f"Optimizing {len(lora_parameters)} LoRA parameter tensors")
  optimizer = AdamW(lora_parameters, lr=args.lr)

  best_val_loss = float('inf')

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    # Training phase
    model.train()
    train_loss = 0
    num_train_batches = 0

    for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_train_batches += 1

    train_loss = train_loss / num_train_batches

    # Validation phase
    model.eval()
    val_loss = 0
    num_val_batches = 0
    
    with torch.no_grad():
      for batch in tqdm(val_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        
        val_loss += loss.item()
        num_val_batches += 1
    
    val_loss = val_loss / num_val_batches
    
    print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, val loss :: {val_loss:.3f}")
    
    # Save best model
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_model(model, optimizer, args, args.filepath)
      print(f"New best validation loss: {best_val_loss:.3f}")
    
    # Generate sample sonnets
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False)

  model = LoRASonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/lora_generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # LoRA-specific arguments
  parser.add_argument("--lora_rank", type=int, default=4,
                     help="LoRA rank (dimensionality of adaptation)")
  parser.add_argument("--lora_alpha", type=float, default=16.0,
                     help="LoRA alpha (scaling parameter)")
  parser.add_argument("--lora_dropout", type=float, default=0.0,
                     help="LoRA dropout rate")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'lora-{args.epochs}-{args.lr}-{args.lora_rank}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)