# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv_custom
import mmengine_custom.fileio as fileio
import numpy as np
from mmcv_custom.transforms import BaseTransform
from mmcv_custom.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv_custom.transforms import LoadImageFromFile

from mmseg_custom.registry import TRANSFORMS
from mmseg_custom.utils import datafrombytes


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv_custom.imfrombytes``.
            See :fun:``mmcv_custom.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine_custom.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv_custom>=2.0.0rc4, mmengine_custom>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Notes:
            Opts about self.reduce_zero_label and "label_map" are done in this func.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # gt_seg_map contains gt_label which is single channel
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadNYUAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by NYU v2 dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv_custom.imfrombytes``.
            See :fun:``mmcv_custom.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine_custom.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv_custom>=2.0.0rc4, mmengine_custom>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Notes:
            Opts about self.reduce_zero_label and "label_map" are done in this func.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if gt_semantic_seg.ndim == 3:
            print(f'found NYU dataset has labels shape of {gt_semantic_seg.shape}')
            assert gt_semantic_seg.shape[2] == 3, f'found gt_semantic_seg has {gt_semantic_seg.shape[2]} dims'
            gt_semantic_seg = gt_semantic_seg[:, :, 0]
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # gt_seg_map contains gt_label which is single channel
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
        # print(results['seg_fields'])

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str



@TRANSFORMS.register_module()
class LoadCarlaAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong here
        label[gt_semantic_seg[:, :, 0] == 128] = 1
        label[gt_semantic_seg[:, :, 1] == 255] = 2  # TODO to solve different net label size?
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadCityroadAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here
        label[(gt_semantic_seg[:, :, 0] == 128)&(gt_semantic_seg[:, :, 2] == 128)] = 1
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadZJUAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        # so far, only cityscape-19c need flag to be color, other datasets remain unchanged.
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='color',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        # gt_seg_map contains gt_label which is single channel
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here        
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 0
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 1
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 2
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 3
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 4
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 5
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 6
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 7
        label[(gt_semantic_seg[:, :, 0] == 64) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 8
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')
        # list = np.array(results['seg_fields'])
        # print(list.shape)
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
    

@TRANSFORMS.register_module()
class LoadZJUAnnotations8(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        # so far, only cityscape-19c need flag to be color, other datasets remain unchanged.
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='color',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        # gt_seg_map contains gt_label which is single channel
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here       
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 0
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 1
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 2
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 3
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 4
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 5
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 128) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 6
        label[(gt_semantic_seg[:, :, 0] == 64) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 7
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')
        # list = np.array(results['seg_fields'])
        # print(list.shape)
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadFMBAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        # so far, only cityscape-19c need flag to be color, other datasets remain unchanged.
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='color',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        # gt_seg_map contains gt_label which is single channel
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here        
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 255
        label[(gt_semantic_seg[:, :, 0] == 179) & (gt_semantic_seg[:, :, 1] == 228) & (
                    gt_semantic_seg[:, :, 2] == 228)] = 0
        label[(gt_semantic_seg[:, :, 0] == 181) & (gt_semantic_seg[:, :, 1] == 57) & (
                    gt_semantic_seg[:, :, 2] == 133)] = 1
        label[(gt_semantic_seg[:, :, 0] == 67) & (gt_semantic_seg[:, :, 1] == 162) & (
                    gt_semantic_seg[:, :, 2] == 177)] = 2
        label[(gt_semantic_seg[:, :, 0] == 200) & (gt_semantic_seg[:, :, 1] == 178) & (
                    gt_semantic_seg[:, :, 2] == 50)] = 3
        label[(gt_semantic_seg[:, :, 0] == 132) & (gt_semantic_seg[:, :, 1] == 45) & (
                    gt_semantic_seg[:, :, 2] == 199)] = 4
        label[(gt_semantic_seg[:, :, 0] == 66) & (gt_semantic_seg[:, :, 1] == 172) & (
                    gt_semantic_seg[:, :, 2] == 84)] = 5
        label[(gt_semantic_seg[:, :, 0] == 179) & (gt_semantic_seg[:, :, 1] == 73) & (
                    gt_semantic_seg[:, :, 2] == 79)] = 6
        label[(gt_semantic_seg[:, :, 0] == 76) & (gt_semantic_seg[:, :, 1] == 99) & (
                    gt_semantic_seg[:, :, 2] == 166)] = 7
        label[(gt_semantic_seg[:, :, 0] == 66) & (gt_semantic_seg[:, :, 1] == 121) & (
                    gt_semantic_seg[:, :, 2] == 253)] = 8
        label[(gt_semantic_seg[:, :, 0] == 137) & (gt_semantic_seg[:, :, 1] == 165) & (
                    gt_semantic_seg[:, :, 2] == 91)] = 9
        label[(gt_semantic_seg[:, :, 0] == 155) & (gt_semantic_seg[:, :, 1] == 97) & (
                    gt_semantic_seg[:, :, 2] == 152)] = 10
        label[(gt_semantic_seg[:, :, 0] == 105) & (gt_semantic_seg[:, :, 1] == 153) & (
                    gt_semantic_seg[:, :, 2] == 140)] = 11
        label[(gt_semantic_seg[:, :, 0] == 222) & (gt_semantic_seg[:, :, 1] == 215) & (
                    gt_semantic_seg[:, :, 2] == 158)] = 12
        label[(gt_semantic_seg[:, :, 0] == 135) & (gt_semantic_seg[:, :, 1] == 113) & (
                    gt_semantic_seg[:, :, 2] == 90)] = 13
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')
        # list = np.array(results['seg_fields'])
        # print(list.shape)
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadCityscapesAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        # so far, only cityscape-19c need flag to be color, other datasets remain unchanged.
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='color',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        # gt_seg_map contains gt_label which is single channel
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here        
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 0
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 64) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 1
        label[(gt_semantic_seg[:, :, 0] == 244) & (gt_semantic_seg[:, :, 1] == 35) & (
                    gt_semantic_seg[:, :, 2] == 232)] = 2
        label[(gt_semantic_seg[:, :, 0] == 70) & (gt_semantic_seg[:, :, 1] == 70) & (
                    gt_semantic_seg[:, :, 2] == 70)] = 3
        label[(gt_semantic_seg[:, :, 0] == 102) & (gt_semantic_seg[:, :, 1] == 102) & (
                    gt_semantic_seg[:, :, 2] == 156)] = 4
        label[(gt_semantic_seg[:, :, 0] == 190) & (gt_semantic_seg[:, :, 1] == 153) & (
                    gt_semantic_seg[:, :, 2] == 153)] = 5
        label[(gt_semantic_seg[:, :, 0] == 153) & (gt_semantic_seg[:, :, 1] == 153) & (
                    gt_semantic_seg[:, :, 2] == 153)] = 6
        label[(gt_semantic_seg[:, :, 0] == 250) & (gt_semantic_seg[:, :, 1] == 170) & (
                    gt_semantic_seg[:, :, 2] == 30)] = 7
        label[(gt_semantic_seg[:, :, 0] == 220) & (gt_semantic_seg[:, :, 1] == 220) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 8
        label[(gt_semantic_seg[:, :, 0] == 107) & (gt_semantic_seg[:, :, 1] == 142) & (
                    gt_semantic_seg[:, :, 2] == 35)] = 9
        label[(gt_semantic_seg[:, :, 0] == 152) & (gt_semantic_seg[:, :, 1] == 251) & (
                    gt_semantic_seg[:, :, 2] == 152)] = 10
        label[(gt_semantic_seg[:, :, 0] == 70) & (gt_semantic_seg[:, :, 1] == 130) & (
                    gt_semantic_seg[:, :, 2] == 180)] = 11
        label[(gt_semantic_seg[:, :, 0] == 220) & (gt_semantic_seg[:, :, 1] == 20) & (
                    gt_semantic_seg[:, :, 2] == 60)] = 12
        label[(gt_semantic_seg[:, :, 0] == 255) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 13
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 142)] = 14
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 70)] = 15
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 60) & (
                    gt_semantic_seg[:, :, 2] == 100)] = 16
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 80) & (
                    gt_semantic_seg[:, :, 2] == 100)] = 17
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 230)] = 18
        label[(gt_semantic_seg[:, :, 0] == 119) & (gt_semantic_seg[:, :, 1] == 11) & (
                    gt_semantic_seg[:, :, 2] == 32)] = 19
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')
        # list = np.array(results['seg_fields'])
        # print(list.shape)
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

@TRANSFORMS.register_module()
class LoadCityscapesAnnotations19(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        # so far, only cityscape-19c need flag to be color, other datasets remain unchanged.
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='color',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        # gt_seg_map contains gt_label which is single channel
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here        
        label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 64) & (
                    gt_semantic_seg[:, :, 2] == 128)] = 0
        label[(gt_semantic_seg[:, :, 0] == 244) & (gt_semantic_seg[:, :, 1] == 35) & (
                    gt_semantic_seg[:, :, 2] == 232)] = 1
        label[(gt_semantic_seg[:, :, 0] == 70) & (gt_semantic_seg[:, :, 1] == 70) & (
                    gt_semantic_seg[:, :, 2] == 70)] = 2
        label[(gt_semantic_seg[:, :, 0] == 102) & (gt_semantic_seg[:, :, 1] == 102) & (
                    gt_semantic_seg[:, :, 2] == 156)] = 3
        label[(gt_semantic_seg[:, :, 0] == 190) & (gt_semantic_seg[:, :, 1] == 153) & (
                    gt_semantic_seg[:, :, 2] == 153)] = 4
        label[(gt_semantic_seg[:, :, 0] == 153) & (gt_semantic_seg[:, :, 1] == 153) & (
                    gt_semantic_seg[:, :, 2] == 153)] = 5
        label[(gt_semantic_seg[:, :, 0] == 250) & (gt_semantic_seg[:, :, 1] == 170) & (
                    gt_semantic_seg[:, :, 2] == 30)] = 6
        label[(gt_semantic_seg[:, :, 0] == 220) & (gt_semantic_seg[:, :, 1] == 220) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 7
        label[(gt_semantic_seg[:, :, 0] == 107) & (gt_semantic_seg[:, :, 1] == 142) & (
                    gt_semantic_seg[:, :, 2] == 35)] = 8
        label[(gt_semantic_seg[:, :, 0] == 152) & (gt_semantic_seg[:, :, 1] == 251) & (
                    gt_semantic_seg[:, :, 2] == 152)] = 9
        label[(gt_semantic_seg[:, :, 0] == 70) & (gt_semantic_seg[:, :, 1] == 130) & (
                    gt_semantic_seg[:, :, 2] == 180)] = 10
        label[(gt_semantic_seg[:, :, 0] == 220) & (gt_semantic_seg[:, :, 1] == 20) & (
                    gt_semantic_seg[:, :, 2] == 60)] = 11
        label[(gt_semantic_seg[:, :, 0] == 255) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 0)] = 12
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 142)] = 13
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 70)] = 14
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 60) & (
                    gt_semantic_seg[:, :, 2] == 100)] = 15
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 80) & (
                    gt_semantic_seg[:, :, 2] == 100)] = 16
        label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
                    gt_semantic_seg[:, :, 2] == 230)] = 17
        label[(gt_semantic_seg[:, :, 0] == 119) & (gt_semantic_seg[:, :, 1] == 11) & (
                    gt_semantic_seg[:, :, 2] == 32)] = 18

        # label[(gt_semantic_seg[:, :, 0] == 128) & (gt_semantic_seg[:, :, 1] == 64) & (
        #             gt_semantic_seg[:, :, 2] == 128)] = 7
        # label[(gt_semantic_seg[:, :, 0] == 244) & (gt_semantic_seg[:, :, 1] == 35) & (
        #             gt_semantic_seg[:, :, 2] == 232)] = 8
        # label[(gt_semantic_seg[:, :, 0] == 70) & (gt_semantic_seg[:, :, 1] == 70) & (
        #             gt_semantic_seg[:, :, 2] == 70)] = 11
        # label[(gt_semantic_seg[:, :, 0] == 102) & (gt_semantic_seg[:, :, 1] == 102) & (
        #             gt_semantic_seg[:, :, 2] == 156)] = 12
        # label[(gt_semantic_seg[:, :, 0] == 190) & (gt_semantic_seg[:, :, 1] == 153) & (
        #             gt_semantic_seg[:, :, 2] == 153)] = 13
        # label[(gt_semantic_seg[:, :, 0] == 153) & (gt_semantic_seg[:, :, 1] == 153) & (
        #             gt_semantic_seg[:, :, 2] == 153)] = 17
        # label[(gt_semantic_seg[:, :, 0] == 250) & (gt_semantic_seg[:, :, 1] == 170) & (
        #             gt_semantic_seg[:, :, 2] == 30)] = 19
        # label[(gt_semantic_seg[:, :, 0] == 220) & (gt_semantic_seg[:, :, 1] == 220) & (
        #             gt_semantic_seg[:, :, 2] == 0)] = 20
        # label[(gt_semantic_seg[:, :, 0] == 107) & (gt_semantic_seg[:, :, 1] == 142) & (
        #             gt_semantic_seg[:, :, 2] == 35)] = 21
        # label[(gt_semantic_seg[:, :, 0] == 152) & (gt_semantic_seg[:, :, 1] == 251) & (
        #             gt_semantic_seg[:, :, 2] == 152)] = 22
        # label[(gt_semantic_seg[:, :, 0] == 70) & (gt_semantic_seg[:, :, 1] == 130) & (
        #             gt_semantic_seg[:, :, 2] == 180)] = 23
        # label[(gt_semantic_seg[:, :, 0] == 220) & (gt_semantic_seg[:, :, 1] == 20) & (
        #             gt_semantic_seg[:, :, 2] == 60)] = 24
        # label[(gt_semantic_seg[:, :, 0] == 255) & (gt_semantic_seg[:, :, 1] == 0) & (
        #             gt_semantic_seg[:, :, 2] == 0)] = 25
        # label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
        #             gt_semantic_seg[:, :, 2] == 142)] = 26
        # label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
        #             gt_semantic_seg[:, :, 2] == 70)] = 27
        # label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 60) & (
        #             gt_semantic_seg[:, :, 2] == 100)] = 28
        # label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 80) & (
        #             gt_semantic_seg[:, :, 2] == 100)] = 31
        # label[(gt_semantic_seg[:, :, 0] == 0) & (gt_semantic_seg[:, :, 1] == 0) & (
        #             gt_semantic_seg[:, :, 2] == 230)] = 32
        # label[(gt_semantic_seg[:, :, 0] == 119) & (gt_semantic_seg[:, :, 1] == 11) & (
        #             gt_semantic_seg[:, :, 2] == 32)] = 33
        
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadKittiAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here
        label[(gt_semantic_seg[:, :, 0] == 255)&(gt_semantic_seg[:, :, 2] == 255)] = 1
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadOrfdAnnotations(MMCV_LoadAnnotations):

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        label = np.zeros(results['ori_shape'][:2], dtype=np.uint8)  # wrong (bug) ever showed up here
        label[gt_semantic_seg[:, :] == 255] = 1
        results['gt_seg_map'] = label
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadBiomedicalImageFromFile(BaseTransform):
    """Load an biomedical mage from file.

    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities, and data type is float32
        if set to_float32 = True, or float64 if decode_backend is 'nifti' and
        to_float32 is False.
    - img_shape
    - ori_shape

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine_custom.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv_custom>=2.0.0rc4, mmengine_custom>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        data_bytes = fileio.get(filename, self.backend_args)
        img = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img[None, ...]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalAnnotation(BaseTransform):
    """Load ``seg_map`` annotation provided by biomedical dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_seg_map': np.ndarray (X, Y, Z) or (Z, Y, X)
        }

    Required Keys:

    - seg_map_path

    Added Keys:

    - gt_seg_map (np.ndarray): Biomedical seg map with shape (Z, Y, X) by
        default, and data type is float32 if set to_float32 = True, or
        float64 if decode_backend is 'nifti' and to_float32 is False.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded seg map to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine_custom.fileio` for details.
            Defaults to None.
            Notes: mmcv_custom>=2.0.0rc4, mmengine_custom>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['seg_map_path'], self.backend_args)
        gt_seg_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_seg_map = gt_seg_map.astype(np.float32)

        if self.decode_backend == 'nifti':
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        if self.to_xyz:
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalData(BaseTransform):
    """Load an biomedical image and annotation from file.

    The loading data format is as the following:

    .. code-block:: python

        {
            'img': np.ndarray data[:-1, X, Y, Z]
            'seg_map': np.ndarray data[-1, X, Y, Z]
        }


    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.
    - img_shape
    - ori_shape

    Args:
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine_custom.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv_custom>=2.0.0rc4, mmengine_custom>=0.2.0 required.
    """

    def __init__(self,
                 with_seg=False,
                 decode_backend: str = 'numpy',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv_custom.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['img_path'], self.backend_args)
        data = datafrombytes(data_bytes, backend=self.decode_backend)
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = data[:-1, :]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]

        if self.with_seg:
            gt_seg_map = data[-1, :]
            if self.decode_backend == 'nifti':
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)

            if self.to_xyz:
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)
            results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerLoader(BaseTransform):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if isinstance(single_input, str):
            inputs = dict(img_path=single_input)
        elif isinstance(single_input, np.ndarray):
            inputs = dict(img=single_input)
        elif isinstance(single_input, dict):
            inputs = single_input
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)
