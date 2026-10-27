"""Read-only inventory of local FAIR-CSAR variants; writes only the chosen JSON report."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_variant(folder):
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    info = {'path': str(folder), 'extensions': dict(Counter(p.suffix.lower() for p in files)),
            'bytes': sum(p.stat().st_size for p in files), 'splits': {}}
    roots = sorted({p.parent.parent for p in files if p.parent.name == 'labels'})
    label_maps = {}
    for root in roots:
        name = root.relative_to(folder).as_posix()
        labels = sorted((root / 'labels').glob('*.txt'))
        images = sorted(p for p in (root / 'images').glob('*') if p.is_file()
                        and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'})
        stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        classes, columns, issues = Counter(), Counter(), Counter()
        examples, objects, empty = [], {}, 0
        for label in labels:
            rows = label.read_text(encoding='utf-8-sig').splitlines()
            content = []
            if not any(row.strip() for row in rows):
                empty += 1
            for line_number, row in enumerate(rows, 1):
                if not row.strip():
                    continue
                parts = row.split()
                columns[str(len(parts))] += 1
                try:
                    values = list(map(float, parts))
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError('nonfinite')
                    if values[0] != int(values[0]) or not 0 <= values[0] <= 4:
                        issues['class_outside_0_to_4'] += 1
                    classes[str(int(values[0]))] += 1
                    if len(values) not in (5, 23, 32):
                        issues['unexpected_column_count'] += 1
                    if len(values) >= 5:
                        if any(not 0 <= v <= 1 for v in values[1:5]):
                            issues['bbox_value_outside_0_to_1'] += 1
                        if values[3] <= 0 or values[4] <= 0:
                            issues['nonpositive_box_size'] += 1
                    if len(values) in (23, 32):
                        step = 2 if len(values) == 23 else 3
                        points = [tuple(values[i:i+2]) for i in range(5, len(values), step)]
                        if any(not 0 <= v <= 1 for point in points for v in point):
                            issues['point_outside_0_to_1'] += 1
                        if len(set(points)) != len(points):
                            issues['objects_with_duplicate_points'] += 1
                    content.append(values)
                except (ValueError, IndexError, OverflowError):
                    issues['invalid_numeric_row'] += 1
                    if len(examples) < 5:
                        examples.append(f'{label.name}:{line_number}')
            objects[label.stem] = content
        label_maps[name] = objects
        sample_images = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                         for p in images[:3]]
        info['splits'][name] = {
            'images': len(images), 'labels': len(labels), 'objects': sum(classes.values()),
            'classes': dict(classes), 'row_columns': dict(columns), 'empty_labels': empty,
            'issues': dict(issues), 'invalid_examples': examples,
            'missing_labels': sorted(stems - label_stems)[:10],
            'missing_labels_count': len(stems - label_stems),
            'missing_images': sorted(label_stems - stems)[:10],
            'missing_images_count': len(label_stems - stems),
            'image_examples': sample_images, 'label_example': objects.get(labels[0].stem) if labels else None,
        }
    split_names = list(label_maps)
    info['cross_split_same_stem'] = {}
    for i, a in enumerate(split_names):
        for b in split_names[i+1:]:
            overlap = sorted(set(label_maps[a]) & set(label_maps[b]))
            info['cross_split_same_stem'][a + ' vs ' + b] = {'count': len(overlap), 'examples': overlap[:10]}
    return info, label_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    variants, maps = {}, {}
    for folder in sorted(args.project.glob('FAIR-CSAR*')):
        if folder.is_dir():
            variants[folder.name], maps[folder.name] = inspect_variant(folder)
            print('Scanned', folder.name, flush=True)
    comparisons = []
    names = list(variants)
    for i, a in enumerate(names):
        a_rows = {k: v for split in maps[a].values() for k, v in split.items()}
        for b in names[i+1:]:
            b_rows = {k: v for split in maps[b].values() for k, v in split.items()}
            shared = a_rows.keys() & b_rows.keys()
            same_boxes = same_points = same_rows = comparable = 0
            for stem in shared:
                ra, rb = a_rows[stem], b_rows[stem]
                if ra == rb:
                    same_rows += 1
                if [r[:5] for r in ra] == [r[:5] for r in rb]:
                    same_boxes += 1
                    if ra and rb and all(len(r) == 23 for r in ra + rb):
                        comparable += 1
                        if all(sorted(zip(x[5::2], x[6::2])) == sorted(zip(y[5::2], y[6::2]))
                               for x, y in zip(ra, rb)):
                            same_points += 1
            comparisons.append({'a': a, 'b': b, 'shared_label_stems': len(shared),
                                'same_rows': same_rows, 'same_boxes_and_object_order': same_boxes,
                                'pointset_comparable_files': comparable,
                                'same_point_sets_ignoring_point_order': same_points})
    archives = {}
    for path in sorted(args.project.glob('FAIR-CSAR*.zip')):
        with zipfile.ZipFile(path) as archive:
            members = [x for x in archive.infolist() if not x.is_dir()]
            archives[path.name] = {'files': len(members), 'bytes_uncompressed': sum(x.file_size for x in members),
                                   'extensions': dict(Counter(Path(x.filename).suffix.lower() for x in members)),
                                   'examples': [x.filename for x in members[:5]]}
    report = {'variants': variants, 'comparisons': comparisons, 'archives': archives,
              'limitations': ['Image SHA-256 covers only three sample files per split; no full image deduplication.',
                              'Point/box comparisons use exact stored numbers and existing object row order.',
                              'Same filenames do not prove image identity or absence of scene leakage.',
                              'Directory names alone do not establish preprocessing provenance.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Report:', args.output)


if __name__ == '__main__':
    main()
