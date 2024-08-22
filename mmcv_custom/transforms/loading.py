# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import mmengine_custom.fileio as fileio
import numpy as np

import mmcv_custom
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)  # default of self.imdecode_backend is cv2
            img = mmcv_custom.imfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:  # Normalization training: restrict pixel values within 0-1
            img = img.astype(np.float32)
            results['img'] = img / 255
            # print('-----------------------------------------------------------------')
            # print('now normalize images to 0-1')
            # print('-----------------------------------------------------------------')
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        else:
            results['img'] = img
            # print('23333')
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadMultimodalImageFromFile(BaseTransform):
    """Load carla multimodal images from file.
       Note:
           it is used for carla dataset, if you change the scene, you should
           modify some options in it.
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load Carla image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            # img that used customfrombytes func is rgb instead of bgr, so it
            # does not need data+preprocessor to translate it into rgb again
            # TODO: check if img here is rgb and if data_preprocessor:bgr2rgb work again
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='unchanged', backend=self.imdecode_backend).astype(np.float32)
            if len(ano.shape) < 3 and self.modality != 'normal':
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3 and ano.dtype == np.float32, \
                'another image must has 3 dims and float32, ' \
               f'even if depth/disp, but found {ano.dtype} and {ano.ndim}'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img / 255
        if self.modality == 'normal':  # in carla dataset, normal img is uint16, multiplied by 65535, so must divided
            results['ano'] = ano / 65535

        elif self.modality == 'depth':
            assert ano.shape[2] == 4, f'depth_encode in CarlaDataset should have 4 channels, but found {ano.shape[2]}'
            scales = np.array([65536.0, 256.0, 1.0, 0]) / (256 ** 3 - 1) * 1000
            in_meters = np.dot(ano, scales).astype(np.float32)
            max_depth = np.max(in_meters)
            min_depth = np.min(in_meters)
            if max_depth > 12.0:
                max_depth = 12.0
                in_meters[in_meters > max_depth] = max_depth
            epsilon = 1e-7
            depth_normalized = (in_meters - min_depth) / (max_depth - min_depth + epsilon)
            depth_normalized_3ch = np.repeat(depth_normalized, 3, axis=2)
            results['ano'] = depth_normalized_3ch
        elif self.modality == 'disp' or self.modality == 'tdisp':
            disp_real = ano.astype(np.float32) / 255
            max_disp = np.max(disp_real)
            min_disp = np.min(disp_real)
            epsilon = 1e-7
            disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
            disp_normalized_3ch = np.repeat(disp_normalized, 3, axis=2)
            results['ano'] = disp_normalized_3ch
        else:
            raise ValueError(f'modality only support normal, depth, disp and tdisp now, not include {self.modality}!')

        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadKittiImageFromFile(BaseTransform):
    """Load kitti multimodal images from file.
       Note:
           it is used for kitti dataset, if you change the scene, you should
           modify some options in it.
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load kitti image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='unchanged', backend=self.imdecode_backend).astype(np.float32)
            if len(ano.shape) < 3 and self.modality != 'normal':
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3 and ano.dtype == np.float32, 'another image must has 3 dims and float32, ' \
                                                              f'even if depth/disp, but found {ano.dtype} and {ano.ndim}'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img / 255
        if self.modality == 'normal':  # in kitti dataset, normal img is uint16, multiplied by 65535, so must divided
            results['ano'] = ano / 65535

        elif self.modality == 'depth': # in Kitti dataset, all samples has max_depth 65535, min_depth 0
            assert ano.shape[2] == 1, f'lidar_depth_2 in KittiDataset should have 1 channels, but found {ano.shape[2]}'
            max_depth = np.max(ano)
            min_depth = np.min(ano)
            epsilon = 1e-7
            depth_normalized = (ano - min_depth) / (max_depth - min_depth + epsilon)
            depth_normalized_3ch = np.repeat(depth_normalized, 3, axis=2)
            results['ano'] = depth_normalized_3ch
        elif self.modality == 'disp' or self.modality == 'tdisp':
            disp_real = ano.astype(np.float32) / 255
            max_disp = np.max(disp_real)
            min_disp = np.min(disp_real)
            epsilon = 1e-7
            disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
            disp_normalized_3ch = np.repeat(disp_normalized, 3, axis=2)
            results['ano'] = disp_normalized_3ch
        else:
            raise ValueError(f'modality only support normal, depth, disp and tdisp now, not include {self.modality}!')

        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadOrfdImageFromFile(BaseTransform):
    """Load Orfd multimodal images from file.
       Note:
           it is used for kitti dataset, if you change the scene, you should
           modify some options in it.
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load kitti image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='unchanged', backend=self.imdecode_backend).astype(np.float32)
            if len(ano.shape) < 3 and self.modality != 'normal':
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3 and ano.dtype == np.float32, 'another image must has 3 dims and float32, ' \
                                                              f'even if depth/disp, but found {ano.dtype} and {ano.ndim}'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img / 255
        if self.modality == 'normal':  # in Orfd dataset, normal img is uint16, multiplied by 65535, so must divided
            results['ano'] = ano / 65535

        elif self.modality == 'depth': # in orfd dataset, all samples has max_depth 65535, min_depth 0
            assert ano.shape[2] == 1, f'depth in Orfd Dataset should have 1 channels, but found {ano.shape[2]}'
            max_depth = np.max(ano)
            min_depth = np.min(ano)
            epsilon = 1e-7
            depth_normalized = (ano - min_depth) / (max_depth - min_depth + epsilon)
            depth_normalized_3ch = np.repeat(depth_normalized, 3, axis=2)
            results['ano'] = depth_normalized_3ch
        else:
            raise ValueError(f'Orfd modality only support normal and depth now, not include {self.modality}!')

        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadCityscapesImageFromFile(BaseTransform):
    """Load Cityscapes multimodal images from file.
       Note:
           it is used for kitti dataset, if you change the scene, you should
           modify some options in it.
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load Cityscapes image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='unchanged', channel_order = 'bgr', backend=self.imdecode_backend)
            if len(ano.shape) < 3 and self.modality != 'normal':
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3, 'another image must has 3 dims, ' \
                                                              f'even if depth/disp, but found {ano.ndim} dims'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img / 255
            if self.modality == 'normal':  # in Cityscapes dataset, normal img is uint8, multiplied by 255, so must divided
                results['ano'] = ano / 255
            # in Cityscapes dataset, all disp samples are tiff files, in which are raw disp
            elif self.modality == 'disp' or self.modality == 'tdisp':
                disp_real = ano.astype(np.float32)
                max_disp = np.max(disp_real)
                min_disp = np.min(disp_real)
                epsilon = 1e-7
                disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
                disp_normalized_3ch = np.repeat(disp_normalized, 3, axis=2)
                results['ano'] = disp_normalized_3ch
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')
        else:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img
            if self.modality == 'normal':  # in Cityscapes dataset, normal img is uint8, multiplied by 255, so must divided
                # for exp in 2023.10.3 mm training: RGB (0-255) Normal (0-1) only, will be deleted after this exp
                results['ano'] = ano
            # in Cityscapes dataset, all disp samples are tiff files, in which are raw disp
            elif self.modality == 'disp' or self.modality == 'tdisp':
                disp_real = ano
                max_disp = np.max(disp_real)
                min_disp = np.min(disp_real)
                epsilon = 1e-7
                disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
                disp_normalized *= 255
                disp_normalized_3ch = np.repeat(disp_normalized, 3, axis=2)
                results['ano'] = disp_normalized_3ch
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')


        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

@TRANSFORMS.register_module()
class LoadCityscapesImageFromFiledisp(BaseTransform):
    """Load Cityscapes multimodal images from file.
       Note:
           it is used for kitti dataset, if you change the scene, you should
           modify some options in it.
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load Cityscapes image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='color', backend=self.imdecode_backend)
            if len(ano.shape) < 3 and self.modality != 'normal':
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3, 'another image must has 3 dims, ' \
                                                              f'even if depth/disp, but found {ano.ndim} dims'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img / 255
            if self.modality == 'normal':  # in Cityscapes dataset, normal img is uint8, multiplied by 255, so must divided
                results['ano'] = ano / 255
            # in Cityscapes dataset, all disp samples are tiff files, in which are raw disp
            elif self.modality == 'disp' or self.modality == 'tdisp':
                disp_real = ano.astype(np.float32)
                max_disp = np.max(disp_real)
                min_disp = np.min(disp_real)
                epsilon = 1e-7
                disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
                # disp_normalized = np.repeat(disp_normalized, 3, axis=2)
                results['ano'] = disp_normalized
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')
        else:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img
            if self.modality == 'normal':  # in Cityscapes dataset, normal img is uint8, multiplied by 255, so must divided
                # for exp in 2023.10.3 mm training: RGB (0-255) Normal (0-1) only, will be deleted after this exp
                results['ano'] = ano
            # in Cityscapes dataset, all disp samples are tiff files, in which are raw disp
            elif self.modality == 'disp' or self.modality == 'tdisp':
                disp_real = ano
                max_disp = np.max(disp_real)
                min_disp = np.min(disp_real)
                epsilon = 1e-7
                disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
                disp_normalized *= 255
                # disp_normalized = np.repeat(disp_normalized, 3, axis=2)
                results['ano'] = disp_normalized
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')


        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadCityscapesImageFromFiledisp4(BaseTransform):
    """Load Cityscapes multimodal images from file.
       Note:
           it is used for kitti dataset, if you change the scene, you should
           modify some options in it.
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load Cityscapes image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='grayscale16',channel_order = 'bgr', backend=self.imdecode_backend)
            # if len(ano.shape) < 3 and self.modality != 'normal':
            #     ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 2, 'another image must has 2 dims, ' \
                                                              f'even if depth/disp, but found {ano.ndim} dims'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img / 255
            if self.modality == 'normal':  # in Cityscapes dataset, normal img is uint8, multiplied by 255, so must divided
                results['ano'] = ano / 255
            # in Cityscapes dataset, all disp samples are tiff files, in which are raw disp
            elif self.modality == 'disp' or self.modality == 'tdisp':
                disp_real = ano.astype(np.float32)
                max_disp = np.max(disp_real)
                min_disp = np.min(disp_real)
                epsilon = 1e-7
                disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
                # disp_normalized = np.repeat(disp_normalized, 3, axis=2)
                results['ano'] = disp_normalized
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')
        else:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img
            if self.modality == 'normal':  # in Cityscapes dataset, normal img is uint8, multiplied by 255, so must divided
                # for exp in 2023.10.3 mm training: RGB (0-255) Normal (0-1) only, will be deleted after this exp
                results['ano'] = ano
            # in Cityscapes dataset, all disp samples are tiff files, in which are raw disp
            elif self.modality == 'disp' or self.modality == 'tdisp':
                disp_real = ano
                max_disp = np.max(disp_real)
                min_disp = np.min(disp_real)
                epsilon = 1e-7
                disp_normalized = (disp_real - min_disp) / (max_disp - min_disp + epsilon)
                disp_normalized *= 255
                # disp_normalized = np.repeat(disp_normalized, 3, axis=2)
                results['ano'] = disp_normalized
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')


        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
    

@TRANSFORMS.register_module()
class LoadNYUImageFromFile(BaseTransform):
    """Load NYU-v2 multimodal images from file.
       Note:
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load MFNet image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='color', backend=self.imdecode_backend)
            if len(ano.shape) < 3 and self.modality != 'depth':
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3, 'another image must has 3 dims, ' \
                                                              f'even if depth/disp, but found {ano.ndim} dims'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img / 255
            if self.modality == 'depth':  # in NYU-Depth v2 dataset, depth img is uint8, 0-255, so must divided
                results['ano'] = ano / 255
            elif self.modality == 'HHA':
                hha_real = ano.astype(np.float32)
                hha_normalized = hha_real / 255
                assert hha_normalized.shape[2] == 3, f'found hha has only {hha_normalized.shape[2]} channels, ' \
                                                     f'but it should not'
                results['ano'] = hha_normalized
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')
        else:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img
            if self.modality == 'depth':  # TODO decide if repeat depth to 3 ch is necessary
                results['ano'] = ano
            elif self.modality == 'HHA':
                results['ano'] = ano
            else:
                raise ValueError(f'modality of NYU dataset only support depth, normal'
                                 f' and hha now, not include {self.modality}!')


        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadMFImageFromFile(BaseTransform):
    """Load MFNet RGB-T images from file.
       Note:
    Required Keys:

    - img_path
    - ano_path
    Modified Keys:

    - img
    - ano
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv_custom.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = True,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None,
                 modality: str = 'normal') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        # add param named modality
        self.modality = modality

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load MFNet image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        anoname = results['ano_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                ano_bytes = file_client.get(anoname)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                ano_bytes = fileio.get(
                    anoname, backend_args=self.backend_args)
            img = mmcv_custom.customfrombytes(
                img_bytes, flag='unchanged', backend=self.imdecode_backend)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            elif len(img.shape) > 3:
                raise ValueError('RGB image has more than 3 dims, but it should not')
            ano = mmcv_custom.customfrombytes(
                ano_bytes, flag='color', backend=self.imdecode_backend)
            if len(ano.shape) < 3:
                ano = np.expand_dims(ano, -1)
            # in case normal img do not have three dims
            assert ano.ndim == 3, 'another image must has 3 dims, ' \
                                                              f'even if thermal, but found {ano.ndim} dims'
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        assert ano is not None, f'failed to load image: {anoname}'
        if self.to_float32:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img / 255
            if self.modality == 'thermal':  # in NYU-Depth v2 dataset, depth img is uint8, 0-255, so must divided
                results['ano'] = ano / 255
            else:
                raise ValueError(f'modality only support normal and disp now, not include {self.modality}!')
        else:
            img = img.astype(np.float32)
            ano = ano.astype(np.float32)
            results['img'] = img
            if self.modality == 'thermal':  # TODO decide if repeat depth to 3 ch is necessary
                results['ano'] = ano
            else:
                raise ValueError(f'modality of MFNet dataset only support thermal'
                                 f' now, not include {self.modality}!')


        results['img_shape'] = img.shape[:2]
        results['ano_shape'] = ano.shape
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"modality='{self.modality}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(BaseTransform):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in key point detection.
                # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
                # means the visibility of this keypoint. n must be equal to the
                # number of keypoint categories.
                'keypoints': [x1, y1, v1, ..., xn, yn, vn]
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4)
             # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # with (x, y, v) order, in np.float32 type.
            'gt_keypoints': np.ndarray(N, NK, 3)
        }

    Required Keys:

    - instances

      - bbox (optional)
      - bbox_label
      - keypoints (optional)

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_seg_map (np.uint8)
    - gt_keypoints (np.float32)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Defaults to False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv_custom.imfrombytes`.
            See :func:`mmcv_custom.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine_custom.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
            self,
            with_bbox: bool = True,
            with_label: bool = True,
            with_seg: bool = False,
            with_keypoints: bool = False,
            imdecode_backend: str = 'cv2',
            file_client_args: Optional[dict] = None,
            *,
            backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                self.file_client_args, results['seg_map_path'])
            img_bytes = file_client.get(results['seg_map_path'])
        else:
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)
        # TODO figure out why this func don't have an map func?
        results['gt_seg_map'] = mmcv_custom.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class StackByChannel(BaseTransform):
    """stack two modality images

    """

    def __init__(self, keys=('img', 'ano')) -> None:
        self.keys = keys

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to stack image.

        Args:
            results (dict): Result dict from
                :class:`mmengine_custom.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        arr_list = [None] * len(self.keys)
        for i, key in enumerate(self.keys):
            arr_list[i] = results[key]
        stack_img = np.dstack(arr_list)
        results['img'] = stack_img
        results['stack_shape'] = stack_img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'

        return repr_str
