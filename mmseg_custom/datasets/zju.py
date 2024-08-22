# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmseg_custom.registry import DATASETS
from .basesegdataset import BaseSegDataset

import mmengine_custom
import mmengine_custom.fileio as fileio


@DATASETS.register_module()
class ZJUDataset(BaseSegDataset):
    """MultiModal Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png', ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' and ``ano_suffix`` is fixed to '_normal.jpg'
    for MMCityscapes dataset.
    """
    METAINFO = dict(
        classes=(
                #  'background', 
                 'building', 'glass', 'car', 'road',
                 'tree', 'sky', 'pedestrian', 'bicycle'),
        palette=[
                #  [0, 0, 0], 
                 [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                 [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]])

    def __init__(self,
                 img_suffix='_image0_0.png',          
                 seg_map_suffix='_image0.png',
                 **kwargs) -> None:
        indices = list(range(8))
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, 
            # indices=indices,
            **kwargs)
