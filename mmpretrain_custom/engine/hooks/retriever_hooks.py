# Copyright (c) OpenMMLab. All rights reserved
import warnings

from mmengine_custom.hooks import Hook
from mmengine_custom.model import is_model_wrapper

from mmpretrain_custom.models import BaseRetriever
from mmpretrain_custom.registry import HOOKS


@HOOKS.register_module()
class PrepareProtoBeforeValLoopHook(Hook):
    """The hook to prepare the prototype in retrievers.

    Since the encoders of the retriever changes during training, the prototype
    changes accordingly. So the `prototype_vecs` needs to be regenerated before
    validation loop.
    """

    def before_val(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if isinstance(model, BaseRetriever):
            if hasattr(model, 'prepare_prototype'):
                model.prepare_prototype()
        else:
            warnings.warn(
                'Only the `mmpretrain_custom.models.retrievers.BaseRetriever` '
                'can execute `PrepareRetrieverPrototypeHook`, but got '
                f'`{type(model)}`')
