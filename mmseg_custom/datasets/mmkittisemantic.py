# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmseg_custom.registry import DATASETS
from .basesegdataset import BaseSegDataset

import mmengine_custom
import mmengine_custom.fileio as fileio


@DATASETS.register_module()
class MMKITTISemanticDataset(BaseSegDataset):
    """MultiModal KITTI Semantic dataset.
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
                 ano_suffix='.png',
                 seg_map_suffix='.png',
                 modality=None,
                 **kwargs) -> None:
        indices = list(range(2))
        self.ano_suffix = ano_suffix
        self.modality = modality
        print(f'use {modality} as another modality.')
        assert self.modality is not None, 'another modality is not setted ' \
                                          'correctly, please modify your config file!'
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, 
            # indices=indices,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load multimodal-annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        modality_path_map = {
            'disp': self.data_prefix.get('disp_path', None),
            # may add tdisp data here future
            'normal': self.data_prefix.get('normal_path', None)
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