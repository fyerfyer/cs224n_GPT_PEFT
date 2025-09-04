# !/usr/bin/env python3

"""
Evaluation code for Quora paraphrase detection.

model_eval_paraphrase is suitable for the dev (and train) dataloaders where the label information is available.
model_test_paraphrase is suitable for the test dataloader where label information is not available.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from sacrebleu.metrics import CHRF
from datasets import (
  SonnetsDataset,
)

TQDM_DISABLE = False


@torch.no_grad()
def model_eval_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  
  # Token IDs for "yes" and "no" as specified in the task
  YES_TOKEN_ID = 8505
  NO_TOKEN_ID = 3919
  
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']
    raw_labels = batch['labels']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    
    # Handle tokenized labels (convert from token IDs to binary labels)
    if raw_labels.dim() > 1:
        label_ids = raw_labels[:, -1].long()  # Take last token of each sequence
    else:
        label_ids = raw_labels.long()
    
    # Convert token IDs to binary labels (same logic as training)
    if label_ids.max() > 1:
        binary_labels = torch.zeros(label_ids.size(0), dtype=torch.long)
        binary_labels[label_ids == YES_TOKEN_ID] = 1  # "yes" -> 1
        binary_labels[label_ids == NO_TOKEN_ID] = 0   # "no" -> 0
        labels = binary_labels
    else:
        labels = label_ids

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_true.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred, y_true, sent_ids


@torch.no_grad()
def model_test_paraphrase(dataloader, model, device):
  model.eval()  # Switch to eval model, will turn off randomness like dropout.
  y_true, y_pred, sent_ids = [], [], []
  for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
    b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['sent_ids']

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask).cpu().numpy()
    preds = np.argmax(logits, axis=1).flatten()

    y_pred.extend(preds)
    sent_ids.extend(b_sent_ids)

  return y_pred, sent_ids


def test_sonnet(
    test_path='predictions/generated_sonnets.txt',
    gold_path='data/TRUE_sonnets_held_out.txt'
):
    chrf = CHRF()

    # get the sonnets
    generated_sonnets = [x[1] for x in SonnetsDataset(test_path)]
    true_sonnets = [x[1] for x in SonnetsDataset(gold_path)]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]

    # compute chrf
    chrf_score = chrf.corpus_score(generated_sonnets, [true_sonnets])
    return float(chrf_score.score)