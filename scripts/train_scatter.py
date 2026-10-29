"""Recorded E01/E02 runs; same v2 splits and frozen E00 base settings."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from train_e00 import (check_dataset_evidence, check_full_protocol, command,
                       make_data_yaml, sha256, write_json)


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_snapshot(run):
    paths = sorted({*REPO.glob('src/*.py'), *REPO.glob('scripts/*.py'),
                    *REPO.glob('configs/*.json')})
    hashes = {}
    for path in paths:
        relative = path.relative_to(REPO)
        target = run / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(path)
    write_json(run / 'local-source-sha256.json', hashes)
    (run / 'source.patch').write_text(command(['git', 'diff', 'HEAD']), encoding='utf-8')
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['E01', 'E02'], required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/root/datasets/faircsar_local_v2'))
    parser.add_argument('--validation-report', type=Path, default=REPO/'outputs/server-data-validation-v2.json')
    parser.add_argument('--weights', type=Path, default=REPO/'weights/pretrained/v8.3.0/yolov8s.pt')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name):
        parser.error('Use a unique simple run name')
    config = json.loads((REPO/'configs/e00_yolov8s.json').read_text())
    spec = json.loads((REPO/'configs/scatter_dcn_v1.json').read_text())
    frozen = REPO/'splits/faircsar_local_v2'
    manifest = check_dataset_evidence(args.data_root, frozen, args.validation_report)
    base_protocol = check_full_protocol(config, args.weights, args.seed, manifest)
    if args.mode == 'full':
        gate = json.loads((REPO/'outputs/scatter_preflight_v1.json').read_text())
        if gate.get('status') != 'passed':
            raise RuntimeError('GPU preflight must pass before full training')
        # Verify exact tested implementation/configuration, not merely its name.
        for name, digest in gate['source_sha256'].items():
            if sha256(REPO/name) != digest:
                raise RuntimeError('Preflight source/config changed: '+name)
        for arm in ('E01', 'E02'):
            smoke = REPO/'runs'/('{}_smoke_s0_001'.format(arm))/'run-record.json'
            evidence = json.loads(smoke.read_text())
            if evidence.get('status') != 'passed' or evidence.get('source_hashes') != gate['training_source_sha256']:
                raise RuntimeError('Both arms need a passing smoke with identical source')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import init_seeds
    from src.scatter_dcn import ScatterModel
    from src.scatter_training import ScatterTrainer
    if (torch.__version__, torchvision.__version__, ultralytics.__version__) != ('2.0.1+cu117', '0.15.2+cu117', '8.3.235'):
        raise RuntimeError('Environment differs from frozen E00')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    run = REPO/'runs'/args.name
    run.mkdir(exist_ok=False, parents=True)
    record = dict(status='preparing', arm=args.arm, seed=args.seed, mode=args.mode,
                  started_utc=now(), git_commit=command(['git', 'rev-parse', 'HEAD']),
                  git_status=command(['git', 'status', '--porcelain']), command=sys.argv,
                  manifest_sha256=manifest, base_protocol=base_protocol, model_protocol=spec,
                  initial_weights_sha256=sha256(args.weights), test_used=False,
                  config_sha256=sha256(REPO/'configs/e00_yolov8s.json'),
                  torch=torch.__version__, torchvision=torchvision.__version__, ultralytics=ultralytics.__version__,
                  gpu=torch.cuda.get_device_name(0))
    write_json(run/'run-record.json', record)
    try:
        record['source_hashes'] = source_snapshot(run)
        package = Path(ultralytics.__file__).parent
        write_json(run/'upstream-source-sha256.json', {str(p.relative_to(package)):sha256(p)
                   for p in sorted(package.rglob('*')) if p.suffix in ('.py', '.yaml') and p.is_file()})
        (run/'pip-freeze.txt').write_text(command([sys.executable, '-m', 'pip', 'freeze'])+'\n')
        (run/'nvidia-smi.txt').write_text(command(['nvidia-smi'])+'\n')
        data_yaml, expected = make_data_yaml(args.data_root, frozen, run, args.mode=='smoke')
        # Same manifest and image contents. Pose directory carries the paired
        # nine-point labels; only paths and label width differ from E00.
        for split in ('train', 'val'):
            listing = run/(split+'-images.txt')
            listing.write_text(listing.read_text().replace('/detect/images/', '/pose/images/'))
        data_yaml.write_text(data_yaml.read_text().replace('/detect"', '/pose"')+'kpt_shape: [9, 3]\n')
        config.update(task='pose', pose=spec['pose'], kobj=spec['kobj'], data=str(data_yaml),
                      model='yolov8s.yaml', project=str(run.parent), name=args.name, exist_ok=True,
                      seed=args.seed, pretrained=True)
        if args.mode == 'smoke':
            config.update(epochs=2, warmup_epochs=0.0)
        write_json(run/'requested-train-args.json', config)
        trainer = ScatterTrainer(overrides=config, guided=args.arm=='E02')
        init_seeds(args.seed, deterministic=True)
        model = ScatterModel(guided=args.arm=='E02')
        pretrained, _ = load_checkpoint(args.weights)
        model.load(pretrained)
        # Audit every shape-compatible pretrained tensor, including buffers.
        state = model.state_dict()
        matched = {n:t for n,t in pretrained.float().state_dict().items() if n in state and t.shape==state[n].shape}
        if not matched or not all(torch.equal(state[n], t) for n,t in matched.items()):
            raise RuntimeError('Common pretrained tensors differ')
        write_json(run/'initialization-audit.json', dict(
            pretrained_common_tensors={n:tensor_hash(t) for n,t in matched.items()},
            all_initial_tensors={n:tensor_hash(t) for n,t in state.items()},
            parameters=sum(p.numel() for p in model.parameters())))
        trainer.model = model

        def audit(t):
            actual = {split:dict(images=len(loader.dataset), objects=sum(len(row['cls']) for row in loader.dataset.labels))
                      for split,loader in [('train', t.train_loader), ('val', t.test_loader)]}
            write_json(run/'loader-counts.json', actual)
            if actual != expected:
                raise RuntimeError('Loader dropped labels/images: '+repr(actual))
            if any(row['keypoints'].shape[1:] != (9,3) for row in t.train_loader.dataset.labels):
                raise RuntimeError('Invalid point shape')
            (run/'train-transforms.txt').write_text(str(t.train_loader.dataset.transforms))
            record.update(status='running', expected_loader_counts=expected)
            write_json(run/'run-record.json', record)
        trainer.add_callback('on_train_start', audit)
        trainer.train()
        rows=list(csv.DictReader((run/'results.csv').open()))
        if len(rows)!=config['epochs'] or any(not math.isfinite(float(v)) for row in rows for k,v in row.items() if 'loss' in k):
            raise RuntimeError('Incomplete epochs or nonfinite losses')
        checkpoint_hashes={n:sha256(run/'weights'/n) for n in ('best.pt','last.pt')}
        image=Path((run/'val-images.txt').read_text().splitlines()[0])
        prediction=YOLO(str(run/'weights/best.pt'), task='pose').predict(str(image),imgsz=1024,device='0',save=False,verbose=False)
        if len(prediction)!=1:
            raise RuntimeError('Saved checkpoint inference failed')
        best=max(rows,key=lambda row:float(row['metrics/mAP50-95(B)']))
        record.update(status='passed',finished_utc=now(), checkpoint_sha256=checkpoint_hashes,
                      best_epoch=int(best['epoch']), val_mAP50=float(best['metrics/mAP50(B)']),
                      val_mAP50_95=float(best['metrics/mAP50-95(B)']),
                      final_revalidation=trainer.metrics, saved_checkpoint_inference='passed')
        write_json(run/'run-record.json', record)
        print('PASS:', run, flush=True)
    except BaseException as error:
        record.update(status='failed',error=type(error).__name__+': '+str(error))
        write_json(run/'run-record.json', record)
        raise


if __name__ == '__main__':
    main()
