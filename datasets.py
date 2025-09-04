# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    # Pre-tokenize yes/no tokens for efficiency
    self.yes_token_id = self.tokenizer.encode('yes')[0]  # Should be 8505
    self.no_token_id = self.tokenizer.encode('no')[0]    # Should be 3919
    
    # Pre-process and cache tokenized data for better performance
    self._preprocess_data()

  def _preprocess_data(self):
    """Pre-tokenize all data for faster training."""
    print("Pre-tokenizing dataset for faster training...")
    self.tokenized_data = []
    
    for i, (sent1, sent2, label, sent_id) in enumerate(self.dataset):
      # Use consistent prompt format (matches test format)
      prompt = f'Is "{sent1}" a paraphrase of "{sent2}"? Answer "yes" or "no": '
      
      # Tokenize prompt
      encoding = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
      token_ids = encoding['input_ids'].squeeze(0)
      attention_mask = encoding['attention_mask'].squeeze(0)
      
      # Convert binary label to token ID (more efficient than tokenizing strings)
      label_token_id = self.yes_token_id if label == 1 else self.no_token_id
      
      self.tokenized_data.append({
        'token_ids': token_ids,
        'attention_mask': attention_mask,
        'label_token_id': label_token_id,
        'sent_id': sent_id
      })
    
    print(f"Pre-tokenized {len(self.tokenized_data)} examples")

  def __len__(self):
    return len(self.tokenized_data)

  def __getitem__(self, idx):
    return self.tokenized_data[idx]

  def collate_fn(self, all_data):
    # Extract pre-tokenized data
    token_ids_list = [x['token_ids'] for x in all_data]
    attention_mask_list = [x['attention_mask'] for x in all_data]
    label_token_ids = [x['label_token_id'] for x in all_data]
    sent_ids = [x['sent_id'] for x in all_data]

    # Pad sequences to same length
    max_len = max(len(ids) for ids in token_ids_list)
    pad_token_id = self.tokenizer.pad_token_id

    # Pad token ids and attention masks
    padded_token_ids = []
    padded_attention_masks = []
    
    for token_ids, attention_mask in zip(token_ids_list, attention_mask_list):
      pad_length = max_len - len(token_ids)
      padded_token_ids.append(torch.cat([token_ids, torch.full((pad_length,), pad_token_id)]))
      padded_attention_masks.append(torch.cat([attention_mask, torch.zeros(pad_length)]))

    batched_data = {
      'token_ids': torch.stack(padded_token_ids),
      'attention_mask': torch.stack(padded_attention_masks),
      'labels': torch.tensor(label_token_ids, dtype=torch.long),  # Single token IDs
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
