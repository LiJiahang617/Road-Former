dataset_type = 'MMMFDataset'
data_root = '/remote-home/jxhuang/data/MF_RGBT'
sample_scale = (640, 480)
train_pipeline = [
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(4, 25)],
        resize_type='ResizeShortestEdge',
        max_size=1280),  # Note: w, h instead of h, w
    dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiModalPhotoMetricDistortion', to_float32=True),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadMFImageFromFile', to_float32=True, modality='thermal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize',
        scale=sample_scale, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
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
            img_path='images/train',
            thermal_path='thermal/train',
            seg_map_path='labels/train'),
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
            img_path='images/val',
            thermal_path='thermal/val',
            seg_map_path='labels/val'),
        pipeline=val_pipeline))
test_dataloader = dict(
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
            img_path='images/test',
            thermal_path='thermal/test',
            seg_map_path='labels/test'),
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
