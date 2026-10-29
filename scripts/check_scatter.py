"""Preflight numerical/gradient/parameter controls; no test-set access."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.torch_utils import init_seeds
from src.scatter_dcn import SamplingDCN, ScatterModel, point_offsets, regular_grid
from src.scatter_training import BoxSelectedPoseMetrics
from train_e00 import sha256, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    device=torch.device(args.device)
    init_seeds(0,deterministic=True)
    checks={}
    x=torch.randn(2,4,11,13,device=device,requires_grad=True)
    weight=torch.randn(3,4,3,3,device=device,requires_grad=True)
    offsets=torch.zeros(2,18,11,13,device=device,requires_grad=True)
    mask=torch.ones(2,9,11,13,device=device,requires_grad=True)
    y=deform_conv2d(x,offsets,weight,padding=(1,1),mask=mask)
    torch.testing.assert_close(y,F.conv2d(x,weight,padding=1),atol=1e-5,rtol=1e-4)
    y.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in (x,weight,offsets,mask))
    checks['zero_offset_conv_equivalence_and_all_operator_gradients']='passed'

    raw=torch.zeros(1,9,3,5,7,device=device)
    raw[:,:,:2]=regular_grid(device,raw.dtype).view(1,9,2,1,1)/2
    assert torch.count_nonzero(point_offsets(raw.flatten(1,2)))==0
    raw[:,0,0]+=1.5
    raw[:,0,1]-=0.25
    delta=point_offsets(raw.flatten(1,2))
    torch.testing.assert_close(delta[:,0],torch.full_like(delta[:,0],-0.5))
    torch.testing.assert_close(delta[:,1],torch.full_like(delta[:,1],3.0))
    assert torch.count_nonzero(delta[:,2:])==0
    # Actual sampler confirms +dx moves right, +dy moves down.
    ramp=torch.arange(49,device=device,dtype=torch.float32).view(1,1,7,7)
    kernel=torch.zeros(1,1,3,3,device=device);kernel[0,0,0,0]=1
    off=torch.zeros(1,18,7,7,device=device);off[:,0]=1;off[:,1]=2
    sampled=deform_conv2d(ramp,off,kernel,padding=(1,1))
    torch.testing.assert_close(sampled[0,0,2,2],ramp[0,0,2,3])
    checks['coordinate_stride_kernel_order_dy_dx']='passed'

    pretrained,_=load_checkpoint(REPO/'weights/pretrained/v8.3.0/yolov8s.pt')
    models=[]
    hashes=[]
    for guided in (False,True):
        init_seeds(0,deterministic=True)
        model=ScatterModel(guided=guided)
        model.load(pretrained)
        models.append(model)
        hashes.append({n:sha256_tensor(t) for n,t in model.state_dict().items()})
    assert hashes[0]==hashes[1]
    common=pretrained.float().state_dict()
    for n,t in models[0].state_dict().items():
        if n in common and t.shape==common[n].shape:
            assert torch.equal(t,common[n]),n
    counts=[sum(p.numel() for p in m.parameters()) for m in models]
    assert counts[0]==counts[1]
    baseline=DetectionModel('yolov8s.yaml',nc=5,verbose=False)
    base_count=sum(p.numel() for p in baseline.parameters())
    checks['identical_initial_tensors_and_common_COCO_tensors']='passed'
    checks['parameters']={'E00':base_count,'E01':counts[0],'E02':counts[1],'extra':counts[0]-base_count}
    images=torch.rand(2,3,128,128,device=device)
    batch={'img':images,'batch_idx':torch.tensor([0,1],device=device),
           'cls':torch.tensor([[0.],[1.]],device=device),
           'bboxes':torch.tensor([[.5,.5,.5,.5],[.5,.5,.4,.4]],device=device),
           'keypoints':torch.tensor([[[.4+.02*i,.4+.02*i,1.] for i in range(9)]]*2,device=device)}
    predictions=[]
    for model in models:
        model.to(device).train()
        model.args=get_cfg(overrides={'pose':12.,'kobj':1.})
        outputs=model(images)
        predictions.append(outputs[0])
        loss,items=model.loss(batch,outputs)
        assert torch.isfinite(loss).all()
        detection_loss=loss[[0,3,4]].sum()
        gradient=torch.autograd.grad(detection_loss,model.model[-1].cv4[0][-1].weight,
                                    retain_graph=True,allow_unused=True)[0]
        if model.guided:
            assert gradient is not None and gradient.abs().sum()>0 and torch.isfinite(gradient).all()
        else:
            assert gradient is None or gradient.abs().sum()==0
        loss.sum().backward()
        for name in ('kernel','offset','mask'):
            grad=getattr(model.model[-1].sampling,name).weight.grad
            assert grad is not None and grad.abs().sum()>0 and torch.isfinite(grad).all(),name
    for a,b in zip(*predictions):
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
    checks['initial_detection_equal_and_guided_detection_gradient_to_points']='passed'
    metric=BoxSelectedPoseMetrics()
    metric.box.all_ap=torch.full((5,10),.3).numpy()
    metric.pose.all_ap=torch.full((5,10),.9).numpy()
    assert abs(metric.fitness-.3)<1e-6
    assert abs(metric.results_dict['fitness']-.3)<1e-6
    checks['checkpoint_fitness_box_only']='passed'
    # Unfused forward timing on identical input; synchronization brackets
    # timing. This is a preflight measurement, not a published speed claim.
    timings={}
    for arm,model in zip(('E01','E02'),models):
        model.eval()
        with torch.no_grad():
            for _ in range(3):model(images)
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(10):model(images)
            if device.type=='cuda':torch.cuda.synchronize()
            timings[arm]=(time.perf_counter()-start)*1000/10
    paths=sorted({*REPO.glob('src/*.py'),*REPO.glob('scripts/*.py'),*REPO.glob('configs/*.json')})
    hashes={str(p.relative_to(REPO)):sha256(p) for p in paths}
    result={'status':'passed','device':str(device),'checks':checks,'timing_ms_batch2_128_fp32_unfused':timings,
            'source_sha256':hashes,'training_source_sha256':hashes,'test_used':False}
    write_json(REPO/'outputs/scatter_preflight_v1.json',result)
    print(json.dumps(result['checks'],indent=2))
    print('PASS: scatter preflight',device,flush=True)


def sha256_tensor(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__=='__main__':main()
