"""Derive paired v2 labels from immutable materialized v1, retaining first duplicate box."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import struct

from freeze_dataset import sha256
from materialize_dataset import read_checksums

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / 'records/FAIRCSAR_DUPLICATE_RESOLUTION_2026-10-28.json'


def retained_bytes(src, task, row, case):
    raw = src.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row[task+'_label_sha256']:
        raise ValueError('v1 label changed: '+str(src))
    if case is None:
        return raw
    if hashlib.sha256(raw).hexdigest() != case[task+'_label_sha256']:
        raise ValueError('Review case label changed: '+str(src))
    lines = raw.decode('utf-8').splitlines()
    keep, drop = case['keep_row_1based']-1, case['drop_row_1based']-1
    if lines[keep].split()[:5] != lines[drop].split()[:5] or lines[keep].split()[0] != '2':
        raise ValueError('Review case no longer has the expected identical class/box')
    if task == 'pose' and lines[keep] == lines[drop]:
        raise ValueError('Expected two distinct point annotations')
    del lines[drop]
    return ('\n'.join(lines)+'\n').encode('utf-8')


def process(item):
    source, target, row, cases = item
    split, stem = row['split'], row['stem']
    if split == 'quarantine':
        return row, None
    case = cases.get(stem)
    updated = row.copy()
    image = source/'detect/images'/split/(stem+'.png')
    paired_image = source/'pose/images'/split/(stem+'.png')
    if not image.is_file() or not os.path.samefile(image, paired_image):
        raise ValueError('v1 paired image identity failed: '+stem)
    for task in ('detect', 'pose'):
        src_label = source/task/'labels'/split/(stem+'.txt')
        output = retained_bytes(src_label, task, row, case)
        updated[task+'_label_sha256'] = hashlib.sha256(output).hexdigest()
        dst_label = target/task/'labels'/split/(stem+'.txt')
        dst_label.write_bytes(output)
        dst_image = target/task/'images'/split/(stem+'.png')
        os.link(image, dst_image)
        if not os.path.samefile(image, dst_image):
            raise ValueError('v2 image does not share source bytes: '+stem)
    if case:
        updated['objects'] = str(int(row['objects'])-1)
    return updated, stem if case else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Materialized faircsar_local_v1')
    parser.add_argument('--output', type=Path, required=True, help='New faircsar_local_v2 directory')
    parser.add_argument('--freeze-output', type=Path, default=REPO/'splits/faircsar_local_v2')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source, target, frozen = args.source.resolve(), args.output.resolve(), args.freeze_output.resolve()
    old = REPO/'splits/faircsar_local_v1'
    if target.exists():
        raise ValueError('Output already exists: '+str(target))
    if source in target.parents or target == source:
        raise ValueError('Output must be outside source dataset')
    read_checksums(old)
    old_manifest_sha = sha256(old/'manifest.csv')
    marker = json.loads((source/'BUILD_COMPLETE.json').read_text(encoding='utf-8'))
    if marker['manifest_sha256'] != old_manifest_sha or marker['image_mode'] != 'hardlink':
        raise ValueError('Expected verified hardlink v1 dataset')
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    cases = {case['stem']: case for case in policy['cases']}
    if len(cases) != 6 or any(case['split'] != 'train' for case in cases.values()):
        raise ValueError('Expected exactly six reviewed training cases')
    with (old/'manifest.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fieldnames, rows = reader.fieldnames, list(reader)
    if any(stem not in {r['stem'] for r in rows if r['split']=='train'} for stem in cases):
        raise ValueError('Reviewed case missing from v1 manifest')
    make_packet = not frozen.exists()
    target.mkdir(parents=True)
    for task in ('detect', 'pose'):
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                (target/task/kind/split).mkdir(parents=True)
    updated, changed = [], set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (new_row, stem) in enumerate(pool.map(process, ((source,target,r,cases) for r in rows)), 1):
            updated.append(new_row)
            if stem:
                changed.add(stem)
            if i % 2000 == 0:
                print('Derived',i,'/',len(rows),flush=True)
    if changed != set(cases):
        raise ValueError('Reviewed case count differs after generation')
    if make_packet:
        frozen.mkdir(parents=True)
        with (frozen/'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(updated)
        metadata = json.loads((old/'metadata.json').read_text(encoding='utf-8'))
        metadata['version'] = 'faircsar_local_v2'
        metadata['parent_manifest_sha256'] = old_manifest_sha
        metadata['duplicate_resolution_canonical_sha256'] = hashlib.sha256(
            json.dumps(policy,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
        metadata.pop('duplicate_resolution_sha256',None)
        metadata['summary']['train']['objects'] -= 6
        metadata['summary']['train']['classes']['2'] -= 6
        metadata['status'] = 'frozen local curated protocol with six duplicate-box point variants resolved'
        (frozen/'metadata.json').write_bytes((json.dumps(metadata,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
        for name in ('train.txt','val.txt','test.txt','quarantine.txt'):
            (frozen/name).write_bytes((old/name).read_bytes())
        names = ('manifest.csv','metadata.json','train.txt','val.txt','test.txt','quarantine.txt')
        (frozen/'SHA256SUMS').write_bytes((''.join(sha256(frozen/name)+'  '+name+'\n' for name in names)).encode('utf-8'))
    else:
        read_checksums(frozen)
        with (frozen/'manifest.csv').open(encoding='utf-8',newline='') as stream:
            expected = list(csv.DictReader(stream))
        if updated != expected:
            raise ValueError('Regenerated v2 manifest differs from committed freeze packet')
    version = json.loads((frozen/'metadata.json').read_text(encoding='utf-8'))['version']
    (target/'BUILD_COMPLETE.json').write_text(json.dumps({
        'version':version,'manifest_sha256':sha256(frozen/'manifest.csv'),
        'source_fingerprint_sha256':marker['source_fingerprint_sha256'],
        'image_mode':'hardlink','parent_manifest_sha256':old_manifest_sha,
        'retained_images_per_task':14660
    },indent=2)+'\n',encoding='utf-8')
    print('Complete:',target,flush=True)


if __name__ == '__main__':
    main()
