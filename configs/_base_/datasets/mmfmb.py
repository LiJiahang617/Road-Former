dataset_type = 'MMFMB'
data_root = '/remote-home/jxhuang/data/FMB'
sample_scale = (800, 600)
train_pipeline = [
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadFMBAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=sample_scale),
    dict(type='RandomCrop', crop_size=(600, 800), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize',
        scale=sample_scale, keep_ratio=True),
    dict(type='LoadFMBAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
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
        modality='thermal',
        data_prefix=dict(
            img_path='train/Visible',
            thermal_path='train/Infrared',
            seg_map_path='train/color'),
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
        modality='thermal',
        data_prefix=dict(
            img_path='test/Visible',
            thermal_path='test/Infrared',
            seg_map_path='test/color'),
        pipeline=val_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
