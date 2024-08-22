dataset_type = 'MMZJUDataset'
data_root = '/remote-home/jxhuang/data/ZJU_RGBP'
sample_scale = (612,512)  
train_pipeline = [
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadZJUAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=sample_scale),
    dict(type='RandomCrop', crop_size=(512, 612), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),    
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(
        type='Resize', scale=sample_scale),
    dict(type='LoadZJUAnnotations', reduce_zero_label=False),
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
        ano_suffix='_aolp.png',
        data_prefix=dict(
            img_path='train/0',
            normal_path='train/aolp2',
            seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        modality='normal',
        ano_suffix='_aolp.png',
        data_prefix=dict(
            img_path='val/0',
            normal_path='val/aolp2',
            seg_map_path='val/label'),
        pipeline=val_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'], ignore_index=0)
test_evaluator = val_evaluator
