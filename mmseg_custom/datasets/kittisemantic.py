# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmseg_custom.registry import DATASETS
from .basesegdataset import BaseSegDataset

import mmengine_custom
import mmengine_custom.fileio as fileio


@DATASETS.register_module()
class KITTISemanticDataset(BaseSegDataset):
    """ KITTI Semantic dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        indices = list(range(2))
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, 
            # indices=indices,
            **kwargs)
