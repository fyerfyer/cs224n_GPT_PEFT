"""
Quick data-split sanity checks: prints counts and overlap examples between train/dev/test.
Usage:
  python3 check_splits.py --train data/quora-train.csv --dev data/quora-dev.csv
"""
import argparse
from datasets import load_paraphrase_data

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='data/quora-train.csv')
parser.add_argument('--dev', default='data/quora-dev.csv')
parser.add_argument('--test', default='data/quora-test-student.csv')
args = parser.parse_args()

train = load_paraphrase_data(args.train)
dev = load_paraphrase_data(args.dev)
try:
    test = load_paraphrase_data(args.test, split='test')
except Exception:
    test = []

print(f"Train examples: {len(train)}")
print(f"Dev examples:   {len(dev)}")
print(f"Test examples:  {len(test)}")

train_ids = set([s[-1] for s in train])
dev_ids = set([s[-1] for s in dev])

common = train_ids.intersection(dev_ids)
print(f"Exact id overlap between train and dev: {len(common)}")
if len(common) > 0:
    print("Examples of overlapping ids (up to 10):")
    for i, sid in enumerate(list(common)[:10]):
        print(sid)

# Optionally show overlapping sentence pairs (rare but useful)
train_map = {s[-1]: (s[0], s[1], s[2]) for s in train}

if len(common) > 0:
    print('\nSample overlapping records:')
    for sid in list(common)[:5]:
        t = train_map.get(sid)
        print(sid, t)

print('\nDone.')
