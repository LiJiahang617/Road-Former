dataset_type = 'MMOrfdDataset'
data_root = '/home/ljh/Desktop/Workspace/Inter_Attention-Network/datasets/orfd'
sample_scale = (640, 352)
train_pipeline = [
    dict(type='LoadOrfdImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadOrfdAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=sample_scale),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadOrfdImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale),
    dict(type='LoadOrfdAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='images/training',
            depth_path='depth/training',
            normal_path='normal/training',
            seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='images/validation',
            depth_path='depth/validation',
            normal_path='normal/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='images/testing',
            depth_path='depth/testing',
            normal_path='normal/testing',
            seg_map_path='annotations/testing'),
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
