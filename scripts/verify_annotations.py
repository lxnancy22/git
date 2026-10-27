"""Read-only label geometry audit and deterministic visual review of FAIR-CSAR.

Requires Pillow. Source images and labels are never modified.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


NAMES = {0: 'Aircraft', 1: 'Ship', 2: 'Oil tank', 3: 'Bridge', 4: 'Tower crane'}


def scene(stem):
    return re.sub(r'_\d+_\d+$', '', stem)


def read_file(item):
    split, path = item
    data = path.read_bytes()
    image_path = path.parent.parent / 'images' / (path.stem + '.png')
    with Image.open(image_path) as image:
        width, height = image.size
        mode = image.mode
    rows, errors = [], []
    for line_no, line in enumerate(data.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            v = list(map(float, line.split()))
            if len(v) != 23 or not all(math.isfinite(x) for x in v):
                raise ValueError('Expected 23 finite values')
            if v[0] != int(v[0]) or int(v[0]) not in NAMES:
                raise ValueError('Unexpected class ID')
            rows.append({'split': split, 'file': path.name, 'line': line_no,
                         'class_id': int(v[0]), 'box': v[1:5],
                         'points': list(zip(v[5::2], v[6::2])),
                         'width': width, 'height': height})
        except ValueError as error:
            errors.append({'split': split, 'file': path.name, 'line': line_no, 'error': str(error)})
    return rows, errors, {'split': split, 'file': path.name,
                          'sha256': hashlib.sha256(data).hexdigest(),
                          'width': width, 'height': height, 'mode': mode, 'objects': len(rows)}


def geometry(row):
    cx, cy, bw, bh = row['box']
    w, h = row['width'], row['height']
    x0, y0, x1, y1 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
    pts = row['points']
    # One source-pixel tolerance for independently rounded box/point coordinates.
    tx, ty = 1/w, 1/h
    flags = []
    if any(not 0 <= x <= 1 for x in row['box']):
        flags.append('box_value_outside_0_1')
    if bw <= 0 or bh <= 0:
        flags.append('nonpositive_box')
    if any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in pts):
        flags.append('point_outside_image')
    if x0 < -tx or y0 < -ty or x1 > 1+tx or y1 > 1+ty:
        flags.append('box_crosses_image_edge')
    if x1 <= 0 or y1 <= 0 or x0 >= 1 or y0 >= 1:
        flags.append('box_disjoint_from_image')
    if any(x < x0-tx or x > x1+tx or y < y0-ty or y > y1+ty for x, y in pts):
        flags.append('point_outside_box_1px_tolerance')
    if len(set(pts)) < 9:
        flags.append('duplicate_point_coordinates')
    px = [x*w for x, _ in pts]
    py = [y*h for _, y in pts]
    mx, my = sum(px)/9, sum(py)/9
    xx = sum((x-mx)**2 for x in px)
    yy = sum((y-my)**2 for y in py)
    xy = sum((x-mx)*(y-my) for x, y in zip(px, py))
    angle = .5 * math.atan2(2*xy, xx-yy)
    projections = [(x-mx)*math.cos(angle)+(y-my)*math.sin(angle) for x, y in zip(px, py)]
    differences = [b-a for a, b in zip(projections, projections[1:])]
    pca_monotone = all(d >= -1 for d in differences) or all(d <= 1 for d in differences)
    angles = [math.atan2(y-my, x-mx) % (2*math.pi) for x, y in zip(px, py)]
    cyclic_diffs = [angles[(i+1) % 9]-angles[i] for i in range(9)]
    angular_cycle = sum(d < -1e-8 for d in cyclic_diffs) <= 1 or sum(d > 1e-8 for d in cyclic_diffs) <= 1
    xy_sorted = pts == sorted(pts)
    return flags, {'pca_monotone_1px': pca_monotone, 'angular_cycle_either_direction': angular_cycle,
                   'xy_lexicographic': xy_sorted}


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def render_case(root, row, output):
    with Image.open(root / row['split'] / 'images' / Path(row['file']).with_suffix('.png')) as source:
        image = source.convert('RGB')
    w, h = image.size
    cx, cy, bw, bh = row['box']
    bx = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
    pts = [(x*w, y*h) for x, y in row['points']]
    left = min(bx[0], *(x for x, _ in pts))
    right = max(bx[2], *(x for x, _ in pts))
    top = min(bx[1], *(y for _, y in pts))
    bottom = max(bx[3], *(y for _, y in pts))
    margin = max(16, .18*max(right-left, bottom-top))
    crop = (math.floor(left-margin), math.floor(top-margin), math.ceil(right+margin), math.ceil(bottom+margin))
    raw = image.crop(crop)
    panel = (560, 420)
    scale = min(panel[0]/raw.width, panel[1]/raw.height)
    size = (max(1, round(raw.width*scale)), max(1, round(raw.height*scale)))
    raw = raw.resize(size, Image.Resampling.NEAREST)
    annotated = raw.copy()
    draw = ImageDraw.Draw(annotated)
    def transform(x, y):
        return (x-crop[0])*scale, (y-crop[1])*scale
    edge = [*transform(0, 0), *transform(w, h)]
    draw.rectangle(edge, outline='#00ffff', width=2)
    draw.rectangle([*transform(bx[0], bx[1]), *transform(bx[2], bx[3])], outline='#00ff66', width=2)
    q = [transform(x, y) for x, y in pts]
    draw.line(q, fill='#efbb44', width=1)
    for i, ((x, y), (ox, oy)) in enumerate(zip(q, pts), 1):
        color = '#ff5555' if not (0 <= ox <= w and 0 <= oy <= h) else '#ffcc00'
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        draw.text((x+4, y-15), str(i), fill=color, font=font(15), stroke_width=1, stroke_fill='black')
    canvas = Image.new('RGB', (1160, 535), '#19212d')
    d = ImageDraw.Draw(canvas)
    d.text((15, 8), f"ID {row['class_id']} ({NAMES[row['class_id']]} candidate) | {row['split']} | row {row['line']}", font=font(18), fill='white')
    d.text((15, 34), Path(row['file']).stem, font=font(12), fill='#c5cbd3')
    for x, pic in [(15, raw), (590, annotated)]:
        canvas.paste(pic, (x+(560-pic.width)//2, 70+(420-pic.height)//2))
    d.text((15, 50), 'Raw context', font=font(14), fill='white')
    d.text((590, 50), 'Box + numbered points', font=font(14), fill='white')
    d.text((15, 499), 'Green: box | Cyan: image edge | Red: off-image point | Yellow line: index path, not physical edges', font=font(14), fill='white')
    canvas.save(output)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--skip-render', action='store_true', help='Refresh statistics while retaining existing previews')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the source dataset')
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(split, p) for split in ('trainval', 'test') for p in sorted((root/split/'labels').glob('*.txt'))]
    rows, errors, manifest = [], [], []
    print(f'Reading {len(tasks)} labels and PNG headers with {args.workers} workers', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (rs, es, item) in enumerate(pool.map(read_file, tasks), 1):
            rows.extend(rs)
            errors.extend(es)
            manifest.append(item)
            if i % 2000 == 0:
                print(f'Read {i}/{len(tasks)} files', flush=True)
    counts, orders, problems = defaultdict(Counter), defaultdict(Counter), []
    rejected = {'detect_derived_5col': set(), 'pose_23col': set()}
    file_objects = Counter((r['split'], r['file']) for r in rows)
    all_keys = set(file_objects)
    by_class, boundary = defaultdict(list), []
    for row in rows:
        flags, diagnostics = geometry(row)
        key = row['split'] + '/class_' + str(row['class_id'])
        counts[key]['objects'] += 1
        counts[key].update(flags)
        orders[key].update({k: int(v) for k, v in diagnostics.items()})
        row['flags'] = flags
        key_file = (row['split'], row['file'])
        # Reproduce guards in the legacy verify_image_label implementation,
        # assuming positive labels have already been parsed and nc=5.
        if any(v < 0 or v > 1 for v in row['box']):
            rejected['detect_derived_5col'].add(key_file)
        if any(v < 0 for v in row['box']) or any(v < 0 or v > 1 for p in row['points'] for v in p):
            rejected['pose_23col'].add(key_file)
        by_class[row['class_id']].append(row)
        if flags:
            problems.append({k: row[k] for k in ('split', 'file', 'line', 'class_id', 'flags', 'box', 'points')})
        if 'box_value_outside_0_1' in flags or 'point_outside_image' in flags:
            boundary.append(row)
    selected = []
    for cid in range(5):
        candidates = [r for r in by_class[cid] if not r['flags'] and r['split'] == 'trainval']
        candidates.sort(key=lambda r: r['box'][2]*r['box'][3])
        used = set()
        cases = []
        for target in (.4, .65, .85):
            start = min(len(candidates)-1, int(len(candidates)*target))
            for row in candidates[start:] + candidates[:start]:
                group = scene(Path(row['file']).stem)
                if group not in used:
                    used.add(group)
                    cases.append(row)
                    break
        sheet = Image.new('RGB', (1160, 535*len(cases)), '#19212d') if not args.skip_render else None
        for i, row in enumerate(cases):
            dest = output/f'class_{cid}_sample_{i+1}.png'
            if sheet is not None:
                sheet.paste(render_case(root, row, dest), (0, 535*i))
            selected.append({**row, 'preview': dest.name})
        if sheet is not None:
            sheet.save(output/f'class_{cid}_sheet.jpg', quality=92)
    for i, row in enumerate(boundary):
        dest = output/f'boundary_{i+1:03d}.jpg'
        if not args.skip_render:
            render_case(root, row, dest)
        row['preview'] = dest.name
    legacy_checks = {}
    for split in ('trainval', 'test'):
        scoped = {key for key in all_keys if key[0] == split}
        d = rejected['detect_derived_5col'] & scoped
        p = rejected['pose_23col'] & scoped
        legacy_checks[split] = {'total_files': len(scoped), 'detect_rejected_files': len(d),
                                'pose_rejected_files': len(p), 'both_rejected_files': len(d & p),
                                'rejected_by_either': len(d | p),
                                'detect_accepted_files': len(scoped-d), 'pose_accepted_files': len(scoped-p),
                                'detect_rejected_objects': sum(file_objects[k] for k in d),
                                'pose_rejected_objects': sum(file_objects[k] for k in p),
                                'common_filter_retained_objects': sum(file_objects[k] for k in scoped-(d|p))}
    def outside_distance(row):
        cx, cy, bw, bh = row['box']
        return max(max((cx-bw/2-x)*row['width'], (x-cx-bw/2)*row['width'],
                       (cy-bh/2-y)*row['height'], (y-cy-bh/2)*row['height'], 0) for x, y in row['points'])
    extra = sorted((r for r in rows if 'point_outside_box_1px_tolerance' in r['flags']
                    and 'point_outside_image' not in r['flags']), key=outside_distance, reverse=True)[:6]
    for i, row in enumerate(extra):
        row['preview'] = f'outside_box_{i+1:02d}.png'
        row['max_outside_box_pixels'] = outside_distance(row)
        if not args.skip_render:
            render_case(root, row, output/row['preview'])
    report = {'source': str(root), 'class_names_status': 'Candidates from legacy sar_pose.yaml; requires semantic review',
              'files': len(manifest), 'objects': len(rows), 'parse_errors': errors,
              'image_headers': dict(Counter(f"{m['width']}x{m['height']} {m['mode']}" for m in manifest)),
              'geometry_by_split_class': dict(counts), 'order_diagnostics': dict(orders),
              'legacy_loader_geometry_simulation': legacy_checks, 'outside_box_examples': extra,
              'typical_samples': selected, 'boundary_cases': boundary,
              'limits': ['Headers read for all images; full pixel decode only for rendered examples.',
                         'Geometric order diagnostics do not establish semantic point identity or sorting provenance.',
                         'Visual samples do not certify every target or official class mapping.']}
    (output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'geometry-problems.json').write_text(json.dumps(problems, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (output/'label-manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'file', 'sha256', 'width', 'height', 'mode', 'objects'])
        writer.writeheader()
        writer.writerows(manifest)
    html = ['<!doctype html><meta charset="utf-8"><title>FAIR-CSAR annotation review</title>',
            '<style>body{background:#19212d;color:white;font:16px sans-serif}img{max-width:100%}article{margin:25px 0}</style>',
            '<h1>FAIR-CSAR annotation review</h1><p>Candidate class names. Source unchanged. Index paths are not physical skeletons.</p>']
    for cid in range(5):
        html.append(f'<article><h2>Class {cid}: {NAMES[cid]} (candidate)</h2><img src="class_{cid}_sheet.jpg"></article>')
    for row in boundary:
        html.append(f'<article><h2>{row["split"]} / {row["file"]} / row {row["line"]}</h2><p>{", ".join(row["flags"])}</p><img loading="lazy" src="{row["preview"]}"></article>')
    for row in extra:
        html.append(f'<article><h2>Outside-box review: {row["file"]} / row {row["line"]}</h2><img loading="lazy" src="{row["preview"]}"></article>')
    (output/'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'Done: {len(rows)} objects, {len(boundary)} boundary cases. Output: {output}', flush=True)


if __name__ == '__main__':
    main()
