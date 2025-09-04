"""
Zero-shot evaluation of pretrained GPT-2 on the Quora dev set using the same
cloze-style prompting and evaluation logic as the training script.

This script mirrors the model->vocab logits -> pick yes/no token pipeline used
by the LoRA/Paraphrase implementations so the results are directly comparable.
"""

import argparse
import torch
import numpy as np
from torch import nn

from datasets import load_paraphrase_data, ParaphraseDetectionDataset
from evaluation import model_eval_paraphrase
from models.gpt2 import GPT2Model
from paraphrase_detection import add_arguments

YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919


class ZeroShotWrapper(nn.Module):
    """Wraps GPT2Model to return binary logits for (no, yes) using vocab logits."""
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model
        self.yes_id = YES_TOKEN_ID
        self.no_id = NO_TOKEN_ID

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids, attention_mask)
        last_token = outputs['last_token']  # [batch, hidden]
        vocab_logits = self.gpt.hidden_state_to_token(last_token)  # [batch, vocab]
        yes_logits = vocab_logits[:, self.yes_id]
        no_logits = vocab_logits[:, self.no_id]
        logits = torch.stack([no_logits, yes_logits], dim=1)
        return logits


def run_zero_shot(model_size='gpt2', batch_size=8, use_gpu=False):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    # Build a minimal args object so add_arguments works
    class A: pass
    args = A()
    args.model_size = model_size
    args = add_arguments(args)

    print(f"Loading pretrained GPT-2 ({model_size})... this may download weights if not cached.")
    gpt = GPT2Model.from_pretrained(model_size, args.d, args.l, args.num_heads)
    gpt = gpt.to(device)
    gpt.eval()

    wrapper = ZeroShotWrapper(gpt).to(device)
    wrapper.eval()

    # Load dev data
    dev_data = load_paraphrase_data('data/quora-dev.csv')
    dev_dataset = ParaphraseDetectionDataset(dev_data, args)
    from torch.utils.data import DataLoader
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

    print("Running zero-shot evaluation on dev set...")
    acc, f1, y_pred, y_true, sent_ids = model_eval_paraphrase(dev_loader, wrapper, device)

    print(f"Zero-shot dev accuracy: {acc:.4f}, f1 (macro): {f1:.4f}")
    return acc, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='gpt2', choices=['gpt2','gpt2-medium','gpt2-large'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    run_zero_shot(model_size=args.model_size, batch_size=args.batch_size, use_gpu=args.use_gpu)
