#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Step 0 - copy raw case folders to ``<dst>/<NNN>_<pinyin>`` and record the mapping.

History: ``filename.py``.  Raw folders were named with Chinese patient names; they are
converted to pinyin and prefixed with a zero-padded index so that later scripts can split a
case id with ``number, name = case.split('_')``.  Case folders must contain the CBCT volume
(``*.nrrd``) and one ``*.mrk.json`` per landmark.

    python scripts/0_standardize_case_names.py --src /data/raw --dst /data/imageStandardData \
        --csv /data/image_data.csv
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='folder with one sub-folder per raw case')
    ap.add_argument('--dst', required=True, help='output folder (imageStandardData)')
    ap.add_argument('--csv', default=None, help='where to save the old->new name table')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--no-pinyin', action='store_true', help='keep the original folder name instead of pinyin')
    args = ap.parse_args()

    if not args.no_pinyin:
        import pypinyin  # only needed for Chinese folder names

    os.makedirs(args.dst, exist_ok=True)
    rows = []
    counter = args.start
    for old_name in sorted(os.listdir(args.src)):
        src_dir = os.path.join(args.src, old_name)
        if not os.path.isdir(src_dir) or not os.listdir(src_dir):
            print('%s is empty or not a folder, skipped - please check!' % old_name)
            continue
        base = old_name if args.no_pinyin else ''.join(pypinyin.lazy_pinyin(old_name))
        base = base.replace('_', '-')            # '_' separates index and name downstream
        new_name = '%03d_%s' % (counter, base)
        counter += 1
        shutil.copytree(src_dir, os.path.join(args.dst, new_name))
        rows.append((new_name, old_name))
        print(old_name, '->', new_name)

    csv_path = args.csv or os.path.join(args.dst, os.pardir, 'image_data.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['standard_name', 'original_name'])
        w.writerows(rows)
    print('mapping saved to', csv_path)


if __name__ == '__main__':
    main()
