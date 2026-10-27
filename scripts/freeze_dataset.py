"""Freeze a local, curated FAIR-CSAR protocol without modifying source files.

Only standard-library dependencies. Full PNG SHA-256 is always computed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re


VERSION = 'faircsar_local_v1'
NAMES = ['Aircraft', 'Ship', 'Oil tank', 'Bridge', 'Tower Crane']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def scene_id(stem):
    match = re.fullmatch(r'(.+)_\d+_\d+', stem)
    if not match:
        raise ValueError(f'Cannot identify source scene: {stem}')
    return match[1]


def location_id(stem):
    match = re.search(r'_([EW][\d.]+)_([NS][\d.]+)_', stem)
    return '_'.join(match.groups()) if match else 'unknown'


def transform_row(text):
    v = [float(x) for x in text.split()]
    if len(v) != 23 or not all(math.isfinite(x) for x in v):
        raise ValueError('Expected 23 finite values')
    if v[0] != int(v[0]) or not 0 <= v[0] <= 4:
        raise ValueError('Class must be an integer in 0..4')
    cid = int(v[0])
    cx, cy, w, h = v[1:5]
    if w <= 0 or h <= 0:
        raise ValueError('Nonpositive source box')
    source = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
    x0, y0, x1, y1 = (max(0., source[0]), max(0., source[1]), min(1., source[2]), min(1., source[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Source box has no positive image intersection')
    box = [(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0]
    box_text = ' '.join(f'{x:.9f}' for x in box)
    det = f'{cid} {box_text}'
    triples = []
    masked = 0
    for x, y in zip(v[5::2], v[6::2]):
        if 0 <= x <= 1 and 0 <= y <= 1:
            # 1 means labelled/in-image, not a claim of optical visibility.
            triples.extend((f'{x:.9f}', f'{y:.9f}', '1'))
        else:
            triples.extend(('0.000000000', '0.000000000', '0'))
            masked += 1
    pose = det + ' ' + ' '.join(triples)
    return det, pose, {'class_id': cid, 'box_clipped': source != (x0, y0, x1, y1), 'masked_points': masked}


def prepare_file(task):
    root, split, path, expected = task
    raw = path.read_bytes()
    label_hash = hashlib.sha256(raw).hexdigest()
    if label_hash != expected:
        raise ValueError(f'Source label changed since annotation audit: {split}/{path.name}')
    detections, poses, counts, clipped, masked = [], [], Counter(), 0, 0
    for line in raw.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        det, pose, info = transform_row(line)
        detections.append(det)
        poses.append(pose)
        counts[str(info['class_id'])] += 1
        clipped += int(info['box_clipped'])
        masked += info['masked_points']
    if not detections:
        raise ValueError(f'Unexpected empty source: {path}')
    image = root/split/'images'/(path.stem+'.png')
    image_hash = sha256(image)
    return {'origin': split, 'stem': path.stem, 'scene': scene_id(path.stem),
            'location': location_id(path.stem), 'source_label_sha256': label_hash,
            'source_image_sha256': image_hash, 'classes': dict(counts), 'objects': len(detections),
            'boxes_clipped': clipped, 'points_masked': masked,
            'det_text': '\n'.join(detections)+'\n', 'pose_text': '\n'.join(poses)+'\n'}


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def assign_partitions(rows, quarantine, seed, ratio, candidates):
    scenes = {r['scene'] for r in rows if r['origin'] == 'trainval'}
    test_scenes = {r['scene'] for r in rows if r['origin'] == 'test'}
    if scenes & test_scenes:
        raise ValueError('Source scene overlap between original trainval and test')
    union = UnionFind(scenes)
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['source_image_sha256']].append(row)
    cross_duplicate_scenes = set()
    duplicates = []
    for h, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        duplicates.append({'sha256': h, 'members': [r['origin']+'/'+r['stem'] for r in group]})
        training = sorted({r['scene'] for r in group if r['origin'] == 'trainval'})
        for s in training[1:]:
            union.union(training[0], s)
        if training and any(r['origin'] == 'test' for r in group):
            cross_duplicate_scenes.update(training)
    # Entire training scene is isolated if an image is byte-identical to source test.
    bad_roots = {union.find(s) for s in cross_duplicate_scenes}
    groups = defaultdict(list)
    for row in rows:
        key = row['origin']+'/'+row['stem']+'.txt'
        if key in quarantine:
            row['split'], row['reason'] = 'quarantine', 'unresolved_class4_bridge_appearance'
        elif row['origin'] == 'trainval' and union.find(row['scene']) in bad_roots:
            row['split'], row['reason'] = 'quarantine', 'source_scene_duplicate_of_original_test'
        elif row['origin'] == 'test':
            row['split'], row['reason'] = 'test', ''
        else:
            row['group'] = union.find(row['scene'])
            groups[row['group']].append(row)
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError('Need at least two training source-scene groups')
    count = min(len(names)-1, max(1, round(len(names)*ratio)))
    group_counts = {}
    total_classes = Counter()
    for name, members in groups.items():
        cls = Counter()
        for row in members:
            cls.update(row['classes'])
        group_counts[name] = len(members), cls
        total_classes.update(cls)
    target_images = sum(len(g) for g in groups.values())*ratio
    rng, best = random.Random(seed), None
    for _ in range(candidates):
        selected = tuple(sorted(rng.sample(names, count)))
        cls, image_count = Counter(), 0
        for name in selected:
            n, c = group_counts[name]
            image_count += n
            cls.update(c)
        remaining = total_classes-cls
        if any(cls[str(k)] == 0 or remaining[str(k)] == 0 for k in range(5)):
            continue
        image_error = abs(image_count-target_images)/target_images
        class_error = sum(abs(cls[str(k)]-ratio*total_classes[str(k)])/(ratio*total_classes[str(k)]) for k in range(5))/5
        score = .4*image_error+.6*class_error
        candidate = (score, selected)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError('No candidate contains all five classes in train and val')
    validation = set(best[1])
    for members in groups.values():
        for row in members:
            row['split'] = 'val' if row['group'] in validation else 'train'
            row['reason'] = ''
    return {'seed': seed, 'target_val_ratio': ratio, 'candidate_count': candidates,
            'algorithm': 'source-scene groups unioned by byte-identical PNG; choose fixed number of groups using seeded candidates and label-only balance objective',
            'validation_group_count': count, 'balance_score': best[0], 'duplicates': duplicates}


def assert_disjoint(rows):
    scenes, hashes, stems = {}, {}, {}
    for split in ('train', 'val', 'test'):
        selected = [r for r in rows if r['split'] == split]
        scenes[split] = {r['scene'] for r in selected}
        hashes[split] = {r['source_image_sha256'] for r in selected}
        stems[split] = {r['stem'] for r in selected}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if scenes[a] & scenes[b] or hashes[a] & hashes[b] or stems[a] & stems[b]:
            raise ValueError(f'Cross-split source scene, exact PNG duplicate, or stem leakage: {a}/{b}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audited-labels', type=Path, required=True)
    parser.add_argument('--quarantine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--val-ratio', type=float, default=.2)
    parser.add_argument('--candidates', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() or root == output or root in output.parents:
        raise ValueError('Freeze output must be a new directory outside source data')
    if not 0 < args.val_ratio < 1:
        raise ValueError('val-ratio must be in (0,1)')
    with args.audited_labels.open(encoding='utf-8', newline='') as f:
        expected = {(r['split'], r['file']): r['sha256'] for r in csv.DictReader(f)}
    quarantine = json.loads(args.quarantine.read_text(encoding='utf-8'))
    quarantine_keys = {r['split']+'/'+r['file'] for r in quarantine}
    actual = [(s, p) for s in ('trainval', 'test') for p in sorted((root/s/'labels').glob('*.txt'))]
    if {(s, p.name) for s, p in actual} != set(expected):
        raise ValueError('Source inventory changed since annotation audit')
    if not quarantine_keys <= {s+'/'+p.name for s, p in actual}:
        raise ValueError('Quarantine includes files absent from source')
    rows = []
    print(f'Hashing all {len(actual)} source PNGs and transforming labels; source untouched', flush=True)
    tasks = [(root, s, p, expected[(s, p.name)]) for s, p in actual]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(prepare_file, tasks), 1):
            rows.append(row)
            if i % 1000 == 0:
                print(f'Hashed {i}/{len(actual)} images', flush=True)
    assignment = assign_partitions(rows, quarantine_keys, args.seed, args.val_ratio, args.candidates)
    assert_disjoint(rows)
    output.mkdir(parents=True)
    summary, changes, manifest = {}, [], []
    for split in ('train', 'val', 'test', 'quarantine'):
        members = sorted((r for r in rows if r['split'] == split), key=lambda r: r['stem'])
        cls = Counter()
        for row in members:
            cls.update(row['classes'])
        summary[split] = {'images': len(members), 'objects': sum(r['objects'] for r in members),
                          'classes': {str(k): cls[str(k)] for k in range(5)},
                          'source_scenes': len({r['scene'] for r in members}),
                          'location_tokens': sorted({r['location'] for r in members}),
                          'boxes_clipped': sum(r['boxes_clipped'] for r in members),
                          'points_masked': sum(r['points_masked'] for r in members)}
        (output/(split+'.txt')).write_text(''.join(f'images/{split}/{r["stem"]}.png\n' for r in members), encoding='utf-8')
        for row in members:
            if split != 'quarantine':
                for task, key in [('detect', 'det_text'), ('pose', 'pose_text')]:
                    dest = output/'labels'/task/split/(row['stem']+'.txt')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(row[key], encoding='utf-8', newline='\n')
            else:
                # Immutable unmodified source text for review, not training labels.
                dest = output/'quarantine_labels'/row['origin']/(row['stem']+'.txt')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((root/row['origin']/'labels'/(row['stem']+'.txt')).read_bytes())
            manifest.append({k: row[k] for k in ['origin', 'stem', 'scene', 'location', 'split', 'reason', 'source_label_sha256', 'source_image_sha256', 'objects', 'boxes_clipped', 'points_masked']})
            if row['boxes_clipped'] or row['points_masked'] or row['reason']:
                changes.append(manifest[-1])
    with (output/'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda r: (r['origin'], r['stem'])))
    payload = '\n'.join(f'{r["origin"]}/{r["stem"]} {r["source_label_sha256"]} {r["source_image_sha256"]}' for r in sorted(rows, key=lambda r: (r['origin'], r['stem'])))
    metadata = {'version': VERSION, 'status': 'frozen local curated protocol; official annotation semantics not fully verified',
                'source_fingerprint_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                'source_root_at_creation': str(root), 'script_sha256': sha256(__file__),
                'names': NAMES, 'pose_shape': [9, 3], 'assignment': assignment, 'summary': summary,
                'policy': {'boxes': 'positive intersection with [0,1]^2; same 9-decimal box text for both tasks',
                           'points': 'original order retained; in-image inclusive [0,1] -> (x,y,1); off-image -> (0,0,0)',
                           'point_flag': 'label validity, not observed physical visibility',
                           'class_recheck': 'source class unchanged; five unresolved source-test images quarantined',
                           'test': 'source-test membership retained except explicit quarantine; this is not the untouched official benchmark',
                           'images': 'no pixel changes; SHA-256 covers every source PNG'},
                'limitations': ['Only source-scene and exact-byte duplicates are blocked across splits; nearby geography and different-date overlap may remain.',
                               'Input is local processed PNG/YOLO data, not verified original SLC/XML.',
                               'Official object-level evidence was unavailable; class names inherit legacy mapping with quarantine.']}
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    checksums = [(p.relative_to(output).as_posix(), sha256(p)) for p in sorted(output.rglob('*')) if p.is_file()]
    (output/'SHA256SUMS').write_text(''.join(h+'  '+name+'\n' for name, h in checksums), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Frozen:', output, flush=True)


if __name__ == '__main__':
    main()
