# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmseg_custom.registry import DATASETS
from .basesegdataset import BaseSegDataset

import mmengine_custom
import mmengine_custom.fileio as fileio


@DATASETS.register_module()
class NYUDataset(BaseSegDataset):
    """SingleModal NYU-v2 Semantic Segmentation 40 categories dataset.
       Original dataset has 894 categories and is mapped into 40 for research.

    The ``img_suffix`` is fixed to '_leftImg8bit.png', ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' and ``ano_suffix`` is fixed to '_normal.jpg'
    for MMCityscapes dataset.
    """
    METAINFO = dict(
        classes=('wall', 'floor', 'cabinet', 'bed', 'chair',
                 'sofa', 'table', 'door', 'window', 'book shelf', 'picture',
                 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow',
                 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel',
                 'shower curtain', 'box', 'white board', 'person', 'night stand', 'toilet', 'sink', 'lamp',
                 'bath tub', 'bag', 'other struct', 'other furntr', 'other prop'),
        palette=[[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                 [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                 [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128], [0, 192, 128], [128, 192, 128],
                 [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128], [192, 64, 128],
                 [64, 192, 128], [192, 192, 128], [0, 0, 64], [128, 0, 64], [0, 128, 64], [128, 128, 64],
                 [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
