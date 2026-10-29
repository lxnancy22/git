"""Matched free-offset and predicted-nine-point DCNv2 experiment heads.

Uses the installed Ultralytics 8.3.235 Detect/Pose implementations (AGPL-3.0).
Point indices retain the frozen annotation order; no geometric reordering.
"""

import torch
from torch import nn
from torchvision.ops import deform_conv2d

from ultralytics.nn.modules.head import Detect, Pose
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8PoseLoss


def regular_grid(device=None, dtype=None):
    """Nine (x,y) kernel displacements, row-major, centered at (0,0)."""
    return torch.tensor([(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)],
                        device=device, dtype=dtype)


def point_offsets(raw):
    """Pose raw x/y -> torchvision interleaved (dy,dx) feature-cell offsets.

    Upstream Pose decodes x_pixel=(2*raw_x+column)*stride. A 3x3,
    padding=1 DCN samples column+kernel_x+offset_x, so offset=2*raw-grid.
    No stride multiplication belongs in the DCN's feature coordinates.
    """
    batch, _, height, width = raw.shape
    xy = raw.reshape(batch, 9, 3, height, width)[:, :, :2] * 2
    grid = regular_grid(raw.device, raw.dtype).view(1, 9, 2, 1, 1)
    return (xy - grid)[:, :, [1, 0]].reshape(batch, 18, height, width)


class SamplingDCN(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.kernel = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.offset = nn.Conv2d(channels, 18, 3, padding=1)
        self.mask = nn.Conv2d(channels, 9, 3, padding=1)
        # Exact identity at initialization, shared by both arms. Multiplying
        # the sigmoid-modulated output by 2 cancels its initial 0.5 mask.
        nn.init.dirac_(self.kernel.weight)
        for predictor in (self.offset, self.mask):
            nn.init.zeros_(predictor.weight)
            nn.init.zeros_(predictor.bias)

    def forward(self, feature, raw_points, guided):
        offset = self.offset(feature)
        if guided:
            offset = offset + point_offsets(raw_points)
        return 2 * deform_conv2d(feature, offset, self.kernel.weight,
                                padding=(1, 1), mask=self.mask(feature).sigmoid())


class ScatterHead(Pose):
    """Coarse points are computed before P3 DCN, never from true labels."""
    def __init__(self, nc=5, ch=(128, 256, 512), guided=False):
        super().__init__(nc=nc, kpt_shape=[9, 3], ch=ch)
        self.guided = guided
        self.sampling = SamplingDCN(ch[0])
        with torch.no_grad():
            for branch in self.cv4:
                nn.init.zeros_(branch[-1].weight)
                bias = branch[-1].bias.view(9, 3)
                bias.zero_()
                bias[:, :2].copy_(regular_grid(dtype=bias.dtype) / 2)

    def forward(self, x):
        batch = x[0].shape[0]
        raw = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpt = torch.cat([p.view(batch, self.nk, -1) for p in raw], -1)
        features = [self.sampling(x[0], raw[0], self.guided), x[1], x[2]]
        detection = Detect.forward(self, features)
        if self.training:
            return detection, kpt
        points = self.kpts_decode(batch, kpt)
        return (torch.cat([detection, points], 1) if self.export else
                (torch.cat([detection[0], points], 1), (detection[1], kpt)))


class ScatterModel(DetectionModel):
    def __init__(self, guided=False, verbose=False):
        super().__init__('yolov8s.yaml', nc=5, verbose=verbose)
        previous = self.model[-1]
        head = ScatterHead(guided=guided)
        # Keep all original Detect tensor names and seed-initialized class
        # tensors. Matching COCO tensors are loaded after construction.
        head.load_state_dict(previous.state_dict(), strict=False)
        for name in ('i', 'f', 'type', 'stride', 'inplace'):
            setattr(head, name, getattr(previous, name))
        head.np = sum(p.numel() for p in head.parameters())
        self.model[-1] = head
        self.kpt_shape = [9, 3]
        self.yaml['kpt_shape'] = [9, 3]
        self.yaml['head'][-1][2:] = ['Pose', ['nc', 'kpt_shape']]
        self.guided = guided

    def init_criterion(self):
        return v8PoseLoss(self)
