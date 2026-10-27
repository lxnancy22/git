"""Package frozen manifests and per-label hashes for Git, without image/label data."""

import argparse
import csv
import json
from pathlib import Path
import shutil

from freeze_dataset import sha256
from materialize_dataset import read_checksums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Frozen packet already exists; use a new version instead of overwriting')
    expected = read_checksums(build)
    with (build/'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for task in ['detect', 'pose']:
            key = f'labels/{task}/{row["split"]}/{row["stem"]}.txt'
            row[task+'_label_sha256'] = '' if row['split'] == 'quarantine' else expected[key]
    output.mkdir(parents=True)
    with (output/'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ['train.txt', 'val.txt', 'test.txt', 'quarantine.txt']:
        shutil.copyfile(build/name, output/name)
    metadata = json.loads((build/'metadata.json').read_text(encoding='utf-8'))
    metadata['packet_format'] = 2
    metadata['label_checksum_storage'] = 'manifest.csv: detect_label_sha256, pose_label_sha256'
    metadata['packaging_script_sha256'] = sha256(__file__)
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'SHA256SUMS').write_text(''.join(sha256(p)+'  '+p.name+'\n' for p in sorted(output.iterdir()) if p.is_file()), encoding='utf-8')
    print('Frozen manifest packet:', output)


if __name__ == '__main__':
    main()
