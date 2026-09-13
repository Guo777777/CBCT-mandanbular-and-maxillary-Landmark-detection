#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 1 - deterministic train / valid / test split (paper: 80 / 10 / 34 of 124 cases).

History: ``1_split_dataset.py``.  Case folders are sorted, shuffled with ``random.seed(2023)``
and written to ``train.txt`` / ``valid.txt`` / ``test.txt`` (one case id per line).

    python scripts/1_split_dataset.py --data-dir /data/imageStandardData --out-dir /data
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='imageStandardData folder (one sub-folder per case)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-train', type=int, default=80)
    ap.add_argument('--n-valid', type=int, default=10)
    ap.add_argument('--seed', type=int, default=2023)
    args = ap.parse_args()

    cases = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    random.seed(args.seed)
    random.shuffle(cases)
    splits = {'train': cases[:args.n_train],
              'valid': cases[args.n_train:args.n_train + args.n_valid],
              'test': cases[args.n_train + args.n_valid:]}

    os.makedirs(args.out_dir, exist_ok=True)
    for name, lst in splits.items():
        path = os.path.join(args.out_dir, name + '.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(''.join(c + '\n' for c in lst))
        print('%s: %d cases -> %s' % (name, len(lst), path))


if __name__ == '__main__':
    main()
