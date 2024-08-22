# Copyright (c) OpenMMLab. All rights reserved.
"""MMDetection provides 17 registry nodes to support using modules across
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

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner', parent=MMENGINE_RUNNERS, locations=['mmdet_custom.engine.runner'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    locations=['mmdet_custom.engine.runner'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop', parent=MMENGINE_LOOPS, locations=['mmdet_custom.engine.runner'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmdet_custom.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mmdet_custom.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['mmdet_custom.datasets.samplers'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmdet_custom.datasets.transforms'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmdet_custom.models'])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmdet_custom.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmdet_custom.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmdet_custom.engine.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmdet_custom.engine.optimizers'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmdet_custom.engine.optimizers'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmdet_custom.engine.schedulers'])
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mmdet_custom.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['mmdet_custom.evaluation'])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['mmdet_custom.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmdet_custom.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmdet_custom.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    # TODO: update the location when mmdet_custom has its own log processor
    locations=['mmdet_custom.engine'])
