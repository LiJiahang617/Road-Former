_base_ = [
   '../_base_/datasets/mmmf_0-1_640x480.py'
]
train_dataloader = dict(
    batch_size=2,
    num_workers=36,)
crop_size = (480, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',    
    mean=[0, 0, 0, 0, 0, 0],
    std=[1, 1, 1, 1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
num_classes = 9
model = dict(
    type='EncoderDecoder_HEA',
    data_preprocessor=data_preprocessor,    
    backbone=dict(
        type='mmpretrain_custom.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.5,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth',
            prefix='backbone.')),      
    neck=dict(
        type='RoadFormer2Neck',
        in_channels=[384, 768, 1536, 3072],
        out_channels=[384, 768, 1536, 3072],
        layer = 4,
        img_scale=crop_size, 
        norm_cfg=dict(type='GN', num_groups=32)
    ),
    decode_head=dict(
        type='RoadFormerHead',
        in_channels=[384, 768, 1536, 3072],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256, 
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet_custom.RoadFormer2PixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet_custom.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet_custom.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet_custom.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet_custom.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet_custom.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet_custom.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet_custom.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet_custom.MaskPseudoSampler')
            )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0)),
        norm_decay_mult=0.0)
    )
max_epochs = 200
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, end_factor=1, by_epoch=False, begin=0, end=1500),   
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=max_epochs,
        by_epoch=True)
]
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=25,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1, draw=False))
default_scope = 'mmseg_custom'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
