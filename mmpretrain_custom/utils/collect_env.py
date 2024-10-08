# Copyright (c) OpenMMLab. All rights reserved.
import mmcv_custom
from mmengine_custom.utils import get_git_hash
from mmengine_custom.utils.dl_utils import collect_env as collect_base_env

import mmpretrain_custom


def collect_env(with_torch_comiling_info=False):
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMCV'] = mmcv_custom.__version__
    if not with_torch_comiling_info:
        env_info.pop('PyTorch compiling details')
    env_info['MMPreTrain'] = mmpretrain_custom.__version__ + '+' + get_git_hash()[:7]
    return env_info
