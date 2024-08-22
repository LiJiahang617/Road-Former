# Copyright (c) OpenMMLab. All rights reserved.
from mmengine_custom.utils import get_git_hash
from mmengine_custom.utils.dl_utils import collect_env as collect_base_env

import mmdet_custom


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet_custom.__version__ + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
