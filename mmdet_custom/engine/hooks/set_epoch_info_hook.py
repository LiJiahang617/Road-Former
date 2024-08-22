# Copyright (c) OpenMMLab. All rights reserved.
from mmengine_custom.hooks import Hook
from mmengine_custom.model.wrappers import is_model_wrapper

from mmdet_custom.registry import HOOKS


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.set_epoch(epoch)
