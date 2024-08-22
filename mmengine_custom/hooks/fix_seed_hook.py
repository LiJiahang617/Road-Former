from mmengine_custom.registry import HOOKS
from .hook import Hook

import torch
import numpy as np
import random

@HOOKS.register_module()
class FixedSeedHook(Hook):
    """Hook for setting a fixed random seed at the beginning of each epoch.

    This is useful for ensuring reproducibility in experiments.
    """

    priority = 'NORMAL'

    def __init__(self, seed=3704):   # 42
        """Initialize the hook with a fixed seed.

        Args:
            seed (int): The fixed seed to use. Defaults to 42.
        """
        self.seed = seed

    def before_train_epoch(self, runner) -> None:
        """Set the fixed seed before each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.set_seed(self.seed)

    def before_val_epoch(self, runner) -> None:
        """Set the fixed seed before each validation epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self.set_seed(self.seed)

    @staticmethod
    def set_seed(seed):
        """Set the random seed for PyTorch, Numpy, and Python random.

        Args:
            seed (int): The seed to set.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False