dataset_type = 'MMCityscapesDataset'
data_root = '/home/jxhuang/data/Cityscapes'
sample_scale = (1024, 512)
train_pipeline = [
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadCityscapesAnnotations', reduce_zero_label=False), 
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize',
        scale=sample_scale),
    dict(type='LoadCityscapesAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/train',
            disp_path='disp/train',
            normal_path='sne/train',
            seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/val',
            disp_path='disp/val',
            normal_path='sne/val',
            seg_map_path='annotations/val'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=16,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_normal.png',
        data_prefix=dict(
            img_path='images/val',
            disp_path='disp/val',
            normal_path='sne/val',
            seg_map_path='annotations/val'),
        pipeline=val_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
