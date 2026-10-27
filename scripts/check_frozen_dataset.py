"""Validate materialized Detect/Pose alignment against a frozen manifest packet."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from freeze_dataset import assert_disjoint, sha256
from materialize_dataset import read_checksums


def validate_pair(item):
    data, row, link_mode = item
    split, stem = row['split'], row['stem']
    raw = {}
    for task in ['detect', 'pose']:
        path = data/task/'labels'/split/(stem+'.txt')
        text = path.read_bytes()
        if hashlib.sha256(text).hexdigest() != row[task+'_label_sha256']:
            raise ValueError('Derived label checksum mismatch: '+str(path))
        raw[task] = [line.split() for line in text.decode().splitlines() if line.strip()]
    if len(raw['detect']) != int(row['objects']) or len(raw['pose']) != len(raw['detect']):
        raise ValueError('Detect/Pose object counts differ: '+stem)
    classes, masked, valid = Counter(), 0, 0
    for det, pose in zip(raw['detect'], raw['pose']):
        if len(det) != 5 or len(pose) != 32 or det != pose[:5]:
            raise ValueError('Detect/Pose prefix or shape mismatch: '+stem)
        v = [float(x) for x in pose]
        if not all(math.isfinite(x) for x in v):
            raise ValueError('Nonfinite derived label: '+stem)
        if int(v[0]) != v[0] or not 0 <= v[0] <= 4:
            raise ValueError('Invalid class: '+stem)
        if any(not 0 <= x <= 1 for x in v[1:5]) or v[3] <= 0 or v[4] <= 0:
            raise ValueError('Invalid derived box: '+stem)
        classes[str(int(v[0]))] += 1
        for i in range(5, 32, 3):
            x, y, flag = v[i:i+3]
            if flag == 0 and x == 0 and y == 0:
                masked += 1
            elif flag == 1 and 0 <= x <= 1 and 0 <= y <= 1:
                valid += 1
            else:
                raise ValueError('Invalid keypoint mask: '+stem)
    a = data/'detect'/'images'/split/(stem+'.png')
    b = data/'pose'/'images'/split/(stem+'.png')
    if link_mode in ['hardlink', 'symlink'] and not os.path.samefile(a, b):
        raise ValueError('Paired image paths do not point to the same content file: '+stem)
    return split, classes, masked, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    read_checksums(args.freeze)
    marker = json.loads((args.data/'BUILD_COMPLETE.json').read_text(encoding='utf-8'))
    if marker['manifest_sha256'] != sha256(args.freeze/'manifest.csv'):
        raise ValueError('Data was built from a different frozen manifest')
    with (args.freeze/'manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert_disjoint(rows)
    metadata = json.loads((args.freeze/'metadata.json').read_text(encoding='utf-8'))
    active = [r for r in rows if r['split'] != 'quarantine']
    for split in ['train', 'val', 'test']:
        names = {r['stem'] for r in active if r['split'] == split}
        listed = (args.freeze/(split+'.txt')).read_text(encoding='utf-8').splitlines()
        if set(listed) != {f'images/{split}/{n}.png' for n in names} or len(listed) != len(names):
            raise ValueError('Frozen list differs from manifest: '+split)
        for task in ['detect', 'pose']:
            for kind, suffix in [('images', '.png'), ('labels', '.txt')]:
                actual = {p.stem for p in (args.data/task/kind/split).glob('*'+suffix)}
                if actual != names:
                    raise ValueError('Missing/extra dataset members: '+task+'/'+kind+'/'+split)
    totals = {s: {'classes': Counter(), 'masked_points': 0, 'valid_points': 0} for s in ['train', 'val', 'test']}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (split, counts, masked, valid) in enumerate(pool.map(validate_pair, [(args.data, r, marker['image_mode']) for r in active]), 1):
            totals[split]['classes'].update(counts)
            totals[split]['masked_points'] += masked
            totals[split]['valid_points'] += valid
            if i % 2000 == 0:
                print(f'Validated {i}/{len(active)} Detect/Pose pairs', flush=True)
    for split, total in totals.items():
        if dict(total['classes']) != metadata['summary'][split]['classes']:
            raise ValueError('Class counts differ from frozen summary: '+split)
        if total['masked_points'] != metadata['summary'][split]['points_masked']:
            raise ValueError('Masked-point counts differ: '+split)
    report = {'version': marker['version'], 'status': 'passed', 'manifest_sha256': marker['manifest_sha256'],
              'source_fingerprint_sha256': marker['source_fingerprint_sha256'],
              'validated_image_pairs': len(active), 'totals': totals,
              'checks': ['frozen packet checksums', 'split lists and exact file inventories',
                         'all derived Detect/Pose label SHA-256', 'identical class/box prefix and object rows',
                         'finite coordinates and valid masks', 'linked image file identity',
                         'cross-split scene, filename and source-PNG hash disjointness'],
              'limits': ['No full training-framework loader run yet.', 'Image bytes were verified during freeze/materialization; this check does not hash all linked images again.']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('PASS:', args.report, flush=True)


if __name__ == '__main__':
    main()
