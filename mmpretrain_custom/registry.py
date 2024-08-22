# Copyright (c) OpenMMLab. All rights reserved.
"""MMPretrain provides 21 registry nodes to support using modules across
projects. Each node is a child of the root registry in mmengine_custom.

More details can be found at
https://mmengine_custom.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine_custom.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine_custom.registry import DATASETS as MMENGINE_DATASETS
from mmengine_custom.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine_custom.registry import HOOKS as MMENGINE_HOOKS
from mmengine_custom.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine_custom.registry import LOOPS as MMENGINE_LOOPS
from mmengine_custom.registry import METRICS as MMENGINE_METRICS
from mmengine_custom.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine_custom.registry import MODELS as MMENGINE_MODELS
from mmengine_custom.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine_custom.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine_custom.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine_custom.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine_custom.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine_custom.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine_custom.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine_custom.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine_custom.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine_custom.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine_custom.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine_custom.registry import Registry

__all__ = [
    'RUNNERS', 'RUNNER_CONSTRUCTORS', 'LOOPS', 'HOOKS', 'LOG_PROCESSORS',
    'OPTIMIZERS', 'OPTIM_WRAPPERS', 'OPTIM_WRAPPER_CONSTRUCTORS',
    'PARAM_SCHEDULERS', 'DATASETS', 'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS',
    'MODEL_WRAPPERS', 'WEIGHT_INITIALIZERS', 'BATCH_AUGMENTS', 'TASK_UTILS',
    'METRICS', 'EVALUATORS', 'VISUALIZERS', 'VISBACKENDS'
]

#######################################################################
#                         mmpretrain_custom.engine                           #
#######################################################################

# Runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner',
    parent=MMENGINE_RUNNERS,
    locations=['mmpretrain_custom.engine'],
)
# Runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    locations=['mmpretrain_custom.engine'],
)
# Loops which define the training or test process, like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop',
    parent=MMENGINE_LOOPS,
    locations=['mmpretrain_custom.engine'],
)
# Hooks to add additional functions during running, like `CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=MMENGINE_HOOKS,
    locations=['mmpretrain_custom.engine'],
)
# Log processors to process the scalar log data.
LOG_PROCESSORS = Registry(
    'log processor',
    parent=MMENGINE_LOG_PROCESSORS,
    locations=['mmpretrain_custom.engine'],
)
# Optimizers to optimize the model weights, like `SGD` and `Adam`.
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmpretrain_custom.engine'],
)
# Optimizer wrappers to enhance the optimization process.
OPTIM_WRAPPERS = Registry(
    'optimizer_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmpretrain_custom.engine'],
)
# Optimizer constructors to customize the hyperparameters of optimizers.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmpretrain_custom.engine'],
)
# Parameter schedulers to dynamically adjust optimization parameters.
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmpretrain_custom.engine'],
)

#######################################################################
#                        mmpretrain_custom.datasets                          #
#######################################################################

# Datasets like `ImageNet` and `CIFAR10`.
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=['mmpretrain_custom.datasets'],
)
# Samplers to sample the dataset.
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['mmpretrain_custom.datasets'],
)
# Transforms to process the samples from the dataset.
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmpretrain_custom.datasets'],
)

#######################################################################
#                         mmpretrain_custom.models                           #
#######################################################################

# Neural network modules inheriting `nn.Module`.
MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    locations=['mmpretrain_custom.models'],
)
# Model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmpretrain_custom.models'],
)
# Weight initialization methods like uniform, xavier.
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmpretrain_custom.models'],
)
# Batch augmentations like `Mixup` and `CutMix`.
BATCH_AUGMENTS = Registry(
    'batch augment',
    locations=['mmpretrain_custom.models'],
)
# Task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util',
    parent=MMENGINE_TASK_UTILS,
    locations=['mmpretrain_custom.models'],
)
# Tokenizer to encode sequence
TOKENIZER = Registry(
    'tokenizer',
    locations=['mmpretrain_custom.models'],
)

#######################################################################
#                       mmpretrain_custom.evaluation                         #
#######################################################################

# Metrics to evaluate the model prediction results.
METRICS = Registry(
    'metric',
    parent=MMENGINE_METRICS,
    locations=['mmpretrain_custom.evaluation'],
)
# Evaluators to define the evaluation process.
EVALUATORS = Registry(
    'evaluator',
    parent=MMENGINE_EVALUATOR,
    locations=['mmpretrain_custom.evaluation'],
)

#######################################################################
#                      mmpretrain_custom.visualization                       #
#######################################################################

# Visualizers to display task-specific results.
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmpretrain_custom.visualization'],
)
# Backends to save the visualization results, like TensorBoard, WandB.
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmpretrain_custom.visualization'],
)
