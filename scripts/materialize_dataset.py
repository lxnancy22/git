"""Build paired Detect/Pose datasets from frozen manifests and unchanged source data.

The freeze directory contains metadata.json, manifest.csv, SHA256SUMS and split lists.
Training labels are regenerated from source, checked against frozen label checksums.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

from freeze_dataset import sha256, transform_row


def read_checksums(freeze):
    entries = {}
    for line in (freeze/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe checksum member path')
        entries[name] = digest
    for name in ['manifest.csv', 'metadata.json', 'train.txt', 'val.txt', 'test.txt', 'quarantine.txt']:
        if sha256(freeze/name) != entries[name]:
            raise ValueError('Frozen file checksum mismatch: '+name)
    return entries


def process_item(task):
    root, output, row, expected, mode = task
    split, stem, origin = row['split'], row['stem'], row['origin']
    label = root/origin/'labels'/(stem+'.txt')
    if sha256(label) != row['source_label_sha256']:
        raise ValueError('Source label checksum mismatch: '+str(label))
    if split == 'quarantine':
        return
    image = root/origin/'images'/(stem+'.png')
    if sha256(image) != row['source_image_sha256']:
        raise ValueError('Source PNG checksum mismatch: '+str(image))
    detect, pose = [], []
    for line in label.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            d, p, _ = transform_row(line)
            detect.append(d)
            pose.append(p)
    for task_name, lines in [('detect', detect), ('pose', pose)]:
        text = '\n'.join(lines)+'\n'
        member = f'labels/{task_name}/{split}/{stem}.txt'
        wanted = row.get(task_name+'_label_sha256') or expected.get(member)
        if wanted is None or hashlib.sha256(text.encode()).hexdigest() != wanted:
            raise ValueError('Regenerated label differs from frozen version: '+member)
        dst_label = output/task_name/'labels'/split/(stem+'.txt')
        dst_label.write_text(text, encoding='utf-8', newline='\n')
        dst_image = output/task_name/'images'/split/(stem+'.png')
        if mode == 'hardlink':
            os.link(image, dst_image)
        elif mode == 'copy':
            shutil.copyfile(image, dst_image)
        else:
            os.symlink(image, dst_image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-mode', choices=['hardlink', 'copy', 'symlink'], default='hardlink')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root, frozen, output = args.root.resolve(), args.freeze.resolve(), args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError('Use a new output directory outside source data; partial outputs are never silently reused')
    expected = read_checksums(frozen)
    metadata = json.loads((frozen/'metadata.json').read_text(encoding='utf-8'))
    with (frozen/'manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    output.mkdir(parents=True)
    for task_name in ['detect', 'pose']:
        for split in ['train', 'val', 'test']:
            for kind in ['images', 'labels']:
                (output/task_name/kind/split).mkdir(parents=True)
    print('Verifying source hashes, regenerating labels and creating paired datasets', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(root, output, row, expected, args.image_mode) for row in rows]
        for i, _ in enumerate(pool.map(process_item, tasks), 1):
            if i % 2000 == 0:
                print(f'Materialized {i}/{len(rows)} manifest rows', flush=True)
    for task_name in ['detect', 'pose']:
        data = output/task_name
        content = [f'path: {json.dumps(data.as_posix())}', 'train: images/train',
                   'val: images/val', 'test: images/test', 'nc: 5', 'names:']
        content += [f'  {i}: {json.dumps(name)}' for i, name in enumerate(metadata['names'])]
        if task_name == 'pose':
            content += ['kpt_shape: [9, 3]', '# v=1 means labelled/in-image; v=0 is masked.',
                        '# No flip_idx: any flip augmentation needs explicitly verified point correspondence.']
        (data/'data.yaml').write_text('\n'.join(content)+'\n', encoding='utf-8')
    (output/'BUILD_COMPLETE.json').write_text(json.dumps({
        'version': metadata['version'], 'source_fingerprint_sha256': metadata['source_fingerprint_sha256'],
        'manifest_sha256': sha256(frozen/'manifest.csv'), 'image_mode': args.image_mode,
        'retained_images_per_task': sum(r['split'] != 'quarantine' for r in rows),
        'image_policy': 'Hardlinks and symlinks share source image content; treat all source and derived images as immutable.'
    }, indent=2)+'\n', encoding='utf-8')
    print('Complete:', output, flush=True)


if __name__ == '__main__':
    main()
