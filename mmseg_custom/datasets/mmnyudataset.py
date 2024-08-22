# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmseg_custom.registry import DATASETS
from .basesegdataset import BaseSegDataset

import mmengine_custom
import mmengine_custom.fileio as fileio


@DATASETS.register_module()
class MMNYUDataset(BaseSegDataset):
    """MultiModal NYU-v2 Semantic Segmentation 40 categories dataset.
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
                 ano_suffix='.png',
                 seg_map_suffix='.png',
                 modality=None,
                 **kwargs) -> None:
        self.ano_suffix = ano_suffix
        self.modality = modality
        print(f'Use {modality} as another modality.')
        assert self.modality is not None, 'another modality is not setted ' \
                                          'correctly, please modify your config file!'
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load multimodal-annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        modality_path_map = {
            'HHA': self.data_prefix.get('hha_path', None),
            # may add tdisp data here future
            'depth': self.data_prefix.get('depth_path', None)
        }
        img_dir = self.data_prefix.get('img_path', None)
        ano_dir = modality_path_map.get(self.modality)
        if ano_dir == None:
            raise ValueError(f'can not find another modality data in '
                             f'{self.modality}, please check your dataset organization!')
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if osp.isfile(self.ann_file):
            lines = mmengine_custom.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                # load rgb info
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                # load another info
                ano = img.replace(self.img_suffix, self.ano_suffix)
                data_info['ano_path'] = osp.join(ano_dir, ano)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list