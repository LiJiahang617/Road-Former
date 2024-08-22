# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine_custom import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmpretrain_custom into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmpretrain_custom default
            scope. If True, the global default scope will be set to
            `mmpretrain_custom`, and all registries will build modules from
            mmpretrain_custom's registry node. To understand more about the registry,
            please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa: E501
    import mmpretrain_custom.datasets  # noqa: F401,F403
    import mmpretrain_custom.engine  # noqa: F401,F403
    import mmpretrain_custom.evaluation  # noqa: F401,F403
    import mmpretrain_custom.models  # noqa: F401,F403
    import mmpretrain_custom.structures  # noqa: F401,F403
    import mmpretrain_custom.visualization  # noqa: F401,F403

    if not init_default_scope:
        return

    current_scope = DefaultScope.get_current_instance()
    if current_scope is None:
        DefaultScope.get_instance('mmpretrain_custom', scope_name='mmpretrain_custom')
    elif current_scope.scope_name != 'mmpretrain_custom':
        warnings.warn(
            f'The current default scope "{current_scope.scope_name}" '
            'is not "mmpretrain_custom", `register_all_modules` will force '
            'the current default scope to be "mmpretrain_custom". If this is '
            'not expected, please set `init_default_scope=False`.')
        # avoid name conflict
        new_instance_name = f'mmpretrain_custom-{datetime.datetime.now()}'
        DefaultScope.get_instance(new_instance_name, scope_name='mmpretrain_custom')
