'''
LoRA-enabled Paraphrase detection.

Running:
  `python lora_paraphrase_detection.py --use_gpu`

trains your LoRAParaphraseGPT model using LoRA fine-tuning and writes the required submission files.
'''

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
from models.lora_gpt2 import LoRAGPT2Model
from optimizer import AdamW

TQDM_DISABLE = False

# Token IDs for "yes" and "no" as specified in the task
YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=5, min_improvement=0.01, verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                          Default: 5
            min_improvement (float): Minimum change in the monitored quantity to qualify as an improvement.
                          Default: 0.01
            verbose (bool): If True, prints a message for each validation loss improvement.
                          Default: True
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.improved = False

    def __call__(self, val_loss):
        """
        Call this method after each validation phase.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should be stopped early
        """
        score = -val_loss
        self.improved = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.improved = True
        elif score < self.best_score + self.min_improvement:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0
            self.improved = True

        return self.early_stop

    def save_checkpoint(self, val_loss):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Model checkpoint updated.')
        self.val_loss_min = val_loss


def seed_everything(seed=11711):
    """Fix the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LoRAParaphraseGPT(nn.Module):
    """
    LoRA-enabled GPT-2 Model designed for cloze-style paraphrase detection.
    
    This model uses LoRA for parameter-efficient fine-tuning on the task of
    determining whether two questions are paraphrases by generating "yes" or "no"
    tokens in response to a formatted prompt.
    """

    def __init__(self, args, lora_config=None):
        super().__init__()
        
        # Default LoRA configuration
        if lora_config is None:
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
        
        # Store token IDs for yes/no classification
        self.yes_token_id = YES_TOKEN_ID
        self.no_token_id = NO_TOKEN_ID
        
        # Print parameter statistics
        self.gpt.print_trainable_parameters()

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
        # Get GPT-2 output
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
    """Train LoRA-enabled GPT-2 for paraphrase detection on the Quora dataset."""
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
    
    # Create LoRA model
    lora_config = {
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
        'dropout': args.lora_dropout
    }
    model = LoRAParaphraseGPT(args, lora_config)
    model = model.to(device)

    # Create optimizer with only LoRA parameters
    lora_parameters = list(model.gpt.get_lora_parameters())
    if not lora_parameters:
        raise ValueError("No LoRA parameters found! Check LoRA implementation.")
    
    print(f"Optimizing {len(lora_parameters)} LoRA parameter tensors")
    optimizer = AdamW(lora_parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience,
                                  min_improvement=args.min_improvement,
                                  verbose=True)

    best_dev_acc = 0
    best_val_loss = float('inf')
    best_epoch = -1

    print(f"Training with early stopping (patience={args.early_stopping_patience}, min_improvement={args.min_improvement})")

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_train_batches = 0
        
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
            num_train_batches += 1

        train_loss = train_loss / num_train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(para_dev_dataloader, desc=f'val-{epoch}', disable=TQDM_DISABLE):
                b_ids = batch['token_ids'].to(device)
                b_mask = batch['attention_mask'].to(device)
                
                # Labels processing for validation loss
                label_ids = batch['labels'].to(device).long()
                binary_labels = torch.zeros(label_ids.size(0), dtype=torch.long, device=device)
                binary_labels[label_ids == YES_TOKEN_ID] = 1  # "yes" -> class 1
                binary_labels[label_ids == NO_TOKEN_ID] = 0   # "no" -> class 0
                labels = binary_labels
                
                logits = model(b_ids, b_mask)
                loss = F.cross_entropy(logits, labels, reduction='mean')
                
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss = val_loss / num_val_batches

        # Evaluate accuracy on dev set
        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        # Check for improvement and save best model
        improvement_status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dev_acc = dev_acc
            best_epoch = epoch
            save_model(model, optimizer, args, args.filepath)
            improvement_status = " ⭐ NEW BEST!"

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, val loss :: {val_loss:.3f}, dev acc :: {dev_acc:.3f}{improvement_status}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.3f} at epoch {best_epoch}")
            break

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.3f} at epoch {best_epoch}")
    print(f"Best dev accuracy: {best_dev_acc:.3f}")


@torch.no_grad()
def test(args):
    """Evaluate LoRA model on dev and test datasets."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load saved model
    saved = torch.load(args.filepath)
    model = LoRAParaphraseGPT(saved['args'])
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
    parser = argparse.ArgumentParser(description="LoRA-enabled paraphrase detection training")

    # Data arguments
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv",
                       help="Path to training data")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv",
                       help="Path to dev data")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv",
                       help="Path to test data")
    parser.add_argument("--para_dev_out", type=str, default="predictions/lora-para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/lora-para-test-output.csv")

    # Training arguments
    parser.add_argument("--seed", type=int, default=11711,
                       help="Random seed")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (recommended: 3-4)")
    parser.add_argument("--use_gpu", action='store_true',
                       help="Use GPU for training")
    parser.add_argument("--batch_size", type=int, default=12,
                       help='Batch size (Tesla T4 safe: 8-16, can try 12-20)')
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (LoRA recommended: 1e-4 to 2e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")

    # Model arguments
    parser.add_argument("--model_size", type=str,
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
                       default='gpt2',
                       help="GPT-2 model size")

    # LoRA-specific arguments (Tesla T4 optimized defaults)
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (recommended: 4-16 for good performance/memory balance)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha parameter (typically rank * 2-4)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate (0.0-0.1 for regularization)")

    # Early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--min_improvement", type=float, default=0.01,
                       help="Minimum change in validation loss to qualify as an improvement")

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
    args.filepath = f'lora-{args.epochs}-{args.lr}-{args.lora_rank}-paraphrase.pt'
    
    print("=" * 70)
    print("🚀 LoRA Fine-tuning for Paraphrase Detection")
    print("=" * 70)
    print(f"Model: {args.model_size}")
    print(f"LoRA Configuration:")
    print(f"  - Rank: {args.lora_rank}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    print(f"Training Parameters:")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - GPU: {args.use_gpu}")
    print(f"Early Stopping:")
    print(f"  - Patience: {args.early_stopping_patience}")
    print(f"  - Min Improvement: {args.min_improvement}")
    print("=" * 70)
    
    seed_everything(args.seed)
    train(args)
    test(args)