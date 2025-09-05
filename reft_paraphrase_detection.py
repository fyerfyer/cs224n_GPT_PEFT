"""
ReFT-enabled Paraphrase detection for GPT-2.

This module provides a ReFT version of the ParaphraseGPT model for parameter-efficient
fine-tuning on the cloze-style paraphrase detection task.
"""

import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.reft_gpt2 import ReFTGPT2Model
from modules.reft import ReFTConfig
from optimizer import AdamW

TQDM_DISABLE = False

# Token IDs for "yes" and "no" as specified in the task
YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919


def seed_everything(seed=11711):
    """Fix the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ReFTParaphraseGPT(nn.Module):
    """
    ReFT-enabled GPT-2 Model designed for cloze-style paraphrase detection.
    
    This model uses ReFT for parameter-efficient fine-tuning on the task of
    determining whether two questions are paraphrases by generating "yes" or "no"
    tokens in response to a formatted prompt.
    """

    def __init__(self, args, reft_config=None):
        super().__init__()
        
        # Default ReFT configuration
        if reft_config is None:
            reft_config = ReFTConfig(
                rank=getattr(args, 'reft_rank', 4),
                alpha=getattr(args, 'reft_alpha', 16.0),
                dropout=getattr(args, 'reft_dropout', 0.0),
                activation=getattr(args, 'reft_activation', 'relu'),
                intervention_locations=getattr(args, 'reft_locations', ['attention']),
                intervention_layers=getattr(args, 'reft_layers', None)  # None means all layers
            )
        
        # Create ReFT-enabled GPT-2 model
        self.gpt = ReFTGPT2Model.from_pretrained(
            model=args.model_size, 
            d=args.d, 
            l=args.l, 
            num_heads=args.num_heads,
            reft_config=reft_config
        )
        
        # Store token IDs for yes/no classification
        self.yes_token_id = YES_TOKEN_ID
        self.no_token_id = NO_TOKEN_ID
        
        # Print parameter statistics
        self.gpt.print_trainable_parameters()
        self.gpt.print_reft_config()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for cloze-style paraphrase detection.
        
        The input is structured as:
        'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
        
        We want to predict the next token, which should be "yes" (token 8505)
        or "no" (token 3919).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits for yes/no classification [batch_size, 2]
        """
        # Get ReFT-enabled GPT-2 output
        outputs = self.gpt(input_ids, attention_mask)
        last_token_hidden = outputs['last_token']  # [batch_size, hidden_size]
        
        # Convert to token logits using weight tying
        all_token_logits = self.gpt.hidden_state_to_token(last_token_hidden)  # [batch_size, vocab_size]
        
        # Extract logits for "yes" and "no" tokens
        yes_logits = all_token_logits[:, self.yes_token_id]  # [batch_size]
        no_logits = all_token_logits[:, self.no_token_id]   # [batch_size]
        
        # Stack to create binary classification logits
        # Index 0 = "no", Index 1 = "yes" (matching expected label format)
        logits = torch.stack([no_logits, yes_logits], dim=1)  # [batch_size, 2]
        
        return logits

    def get_reft_parameters(self):
        """Get only ReFT parameters for optimization."""
        return self.gpt.get_reft_parameters()


def save_model(model, optimizer, args, filepath):
    """Save model checkpoint."""
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saved model to {filepath}")


def train(args):
    """Train ReFT-enabled GPT-2 for paraphrase detection on the Quora dataset."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Create datasets and dataloaders
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn
    )
    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )

    # Add model size arguments
    args = add_arguments(args)
    
    # Create ReFT configuration
    reft_config = ReFTConfig(
        rank=args.reft_rank,
        alpha=args.reft_alpha,
        dropout=args.reft_dropout,
        activation=args.reft_activation,
        intervention_locations=args.reft_locations,
        intervention_layers=args.reft_layers
    )
    
    # Create ReFT model
    model = ReFTParaphraseGPT(args, reft_config)
    model = model.to(device)

    # Create optimizer with only ReFT parameters
    reft_parameters = list(model.get_reft_parameters())
    if not reft_parameters:
        raise ValueError("No ReFT parameters found! Check ReFT implementation.")
    
    print(f"Optimizing {len(reft_parameters)} ReFT parameter tensors")
    optimizer = AdamW(reft_parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    best_dev_acc = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Get batch data
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            
            # Labels are now single token IDs from optimized dataset
            label_ids = batch['labels'].to(device).long()
            
            # Convert token IDs to binary class labels for cross-entropy loss
            binary_labels = torch.zeros(label_ids.size(0), dtype=torch.long, device=device)
            binary_labels[label_ids == YES_TOKEN_ID] = 1  # "yes" -> class 1
            binary_labels[label_ids == NO_TOKEN_ID] = 0   # "no" -> class 0
            labels = binary_labels

            # Forward pass
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, labels, reduction='mean')
            
            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        # Evaluate on dev set
        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}")


@torch.no_grad()
def test(args):
    """Evaluate ReFT model on dev and test datasets."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load saved model
    saved = torch.load(args.filepath)
    model = ReFTParaphraseGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model to test from {args.filepath}")

    # Load datasets
    para_dev_data = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split='test')

    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )
    para_test_dataloader = DataLoader(
        para_test_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_test_data.collate_fn
    )

    # Evaluate
    dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(
        para_dev_dataloader, model, device
    )
    print(f"Dev paraphrase acc :: {dev_para_acc:.3f}")
    
    test_para_y_pred, test_para_sent_ids = model_test_paraphrase(
        para_test_dataloader, model, device
    )

    # Save predictions
    with open(args.para_dev_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
            f.write(f"{p}, {s} \n")

    with open(args.para_test_out, "w+") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(test_para_sent_ids, test_para_y_pred):
            f.write(f"{p}, {s} \n")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/reft-para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/reft-para-test-output.csv")

    # Training arguments
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8, 
                       help='Batch size (8 can fit a 12GB GPU)')
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate (higher for ReFT)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")

    # Model arguments
    parser.add_argument("--model_size", type=str,
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large'], 
                       default='gpt2',
                       help="GPT-2 model size")

    # ReFT-specific arguments
    parser.add_argument("--reft_rank", type=int, default=4,
                       help="ReFT rank (dimensionality of intervention)")
    parser.add_argument("--reft_alpha", type=float, default=16.0,
                       help="ReFT alpha (scaling parameter)")
    parser.add_argument("--reft_dropout", type=float, default=0.0,
                       help="ReFT dropout rate")
    parser.add_argument("--reft_activation", type=str, default='relu',
                       choices=['relu', 'gelu', 'tanh', 'linear'],
                       help="ReFT activation function")
    parser.add_argument("--reft_locations", type=str, nargs='+', 
                       default=['attention'], 
                       choices=['attention', 'ffn'],
                       help="ReFT intervention locations")
    parser.add_argument("--reft_layers", type=int, nargs='+', default=None,
                       help="Specific layers for ReFT interventions (None = all layers)")

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add model size-dependent arguments."""
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
    args.filepath = f'reft-{args.epochs}-{args.lr}-{args.reft_rank}-paraphrase.pt'
    seed_everything(args.seed)
    train(args)
    test(args)