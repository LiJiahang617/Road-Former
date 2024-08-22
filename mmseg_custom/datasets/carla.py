# Copyright (c) OpenMMLab. All rights reserved.
from mmseg_custom.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CarlaDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('background', 'road', 'pothole'),
        palette=[[0, 0, 0], [128, 64, 128], [0, 255, 0]]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
