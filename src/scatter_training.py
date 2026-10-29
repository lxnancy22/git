"""Pose labels/loss with the frozen E00 Box-only checkpoint selection rule."""

from copy import copy

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils.metrics import PoseMetrics

from src.scatter_dcn import ScatterModel


class BoxSelectedPoseMetrics(PoseMetrics):
    @property
    def fitness(self):
        return self.box.fitness()


class BoxSelectedPoseValidator(PoseValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = BoxSelectedPoseMetrics()


class ScatterTrainer(PoseTrainer):
    def __init__(self, *args, guided=False, **kwargs):
        self.guided = guided
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ScatterModel(guided=self.guided, verbose=verbose)
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return BoxSelectedPoseValidator(self.test_loader, save_dir=self.save_dir,
                                        args=copy(self.args), _callbacks=self.callbacks)
