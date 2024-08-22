# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseTransform
from .builder import TRANSFORMS
from .loading import LoadAnnotations, LoadImageFromFile, LoadMultimodalImageFromFile, \
                     StackByChannel, LoadKittiImageFromFile, LoadOrfdImageFromFile, \
                     LoadCityscapesImageFromFile, LoadNYUImageFromFile, LoadMFImageFromFile,LoadCityscapesImageFromFiledisp,LoadCityscapesImageFromFiledisp4
from .processing import (CenterCrop, MultiScaleFlipAug, Normalize, Pad,
                         RandomChoiceResize, RandomFlip, RandomGrayscale,
                         RandomResize, Resize, TestTimeAug)
from .wrappers import (Compose, KeyMapper, RandomApply, RandomChoice,
                       TransformBroadcaster)

try:
    import torch  # noqa: F401
except ImportError:
    __all__ = [
        'BaseTransform', 'TRANSFORMS', 'TransformBroadcaster', 'Compose',
        'RandomChoice', 'KeyMapper', 'LoadImageFromFile', 'LoadMultimodalImageFromFile',
        'LoadAnnotations','StackByChannel', 'LoadKittiImageFromFile',
        'Normalize', 'Resize', 'Pad', 'RandomFlip', 'RandomChoiceResize',
        'CenterCrop', 'RandomGrayscale', 'MultiScaleFlipAug', 'RandomResize',
        'RandomApply', 'TestTimeAug', 'LoadCityscapesImageFromFile','LoadCityscapesImageFromFiledisp','LoadCityscapesImageFromFiledisp4',
        'LoadOrfdImageFromFile', 'LoadNYUImageFromFile', 'LoadMFImageFromFile'
    ]
else:
    from .formatting import ImageToTensor, ToTensor, to_tensor

    __all__ = [
        'BaseTransform', 'TRANSFORMS', 'TransformBroadcaster', 'Compose',
        'RandomChoice', 'KeyMapper', 'LoadImageFromFile', 'LoadAnnotations',
        'Normalize', 'Resize', 'Pad', 'ToTensor', 'to_tensor', 'ImageToTensor',
        'RandomFlip', 'RandomChoiceResize', 'CenterCrop', 'RandomGrayscale',
        'MultiScaleFlipAug', 'RandomResize', 'RandomApply', 'TestTimeAug',
        'StackByChannel', 'LoadKittiImageFromFile', 'LoadMultimodalImageFromFile',
        'LoadCityscapesImageFromFile', 'LoadOrfdImageFromFile', 'LoadNYUImageFromFile',
        'LoadMFImageFromFile','LoadCityscapesImageFromFiledisp','LoadCityscapesImageFromFiledisp4'
    ]
