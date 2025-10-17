"""
BAN (Bi-temporal Adapter Network) Configuration for Ca Mau Forest Change Detection
Author: Ninh Hai Dang (21021411)
Date: 2025-10-17

Model: BAN with ViT-B/16 CLIP encoder + MiT-B0 side encoder
Input: 9 channels per time step (S2 + S1: B4, B8, B11, B12, NDVI, NBR, NDMI, VH, Ratio)
Dataset: Ca Mau forest change (1,285 samples: 1,028 train, 128 val, 129 test)
"""

_base_ = '../open-cd/configs/_base_/default_runtime.py'

# ============================================================================
# Dataset Configuration
# ============================================================================
dataset_type = 'LEVIR_CD_Dataset'
data_root = 'data/processed'
crop_size = (256, 256)

# Training pipeline
train_pipeline = [
    dict(type='MultiImgLoadRasterioFromFile'),  # Custom loader for 9-channel TIFF
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # Note: PhotoMetricDistortion removed - not compatible with 9-channel images
    dict(type='MultiImgPackSegInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='MultiImgLoadRasterioFromFile'),  # Custom loader for 9-channel TIFF
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

# Dataloaders
train_dataloader = dict(
    batch_size=4,  # Adjust based on GPU memory
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='train/label',
            img_path_from='train/A',
            img_path_to='train/B'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='val/label',
            img_path_from='val/A',
            img_path_to='val/B'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            seg_map_path='test/label',
            img_path_from='test/A',
            img_path_to='test/B'),
        img_suffix='.tif',
        seg_map_suffix='.png',
        pipeline=test_pipeline))

# Evaluators
val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])

# ============================================================================
# Model Configuration
# ============================================================================
norm_cfg = dict(type='SyncBN', requires_grad=True)

# Data preprocessor for 9-channel input (9 channels × 2 time steps = 18 channels)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[0.5] * 9 * 2,  # 18 channels: normalized to [0, 1] already
    std=[0.5] * 9 * 2,
    bgr_to_rgb=False,  # Already in correct order
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

checkpoint_side = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

model = dict(
    type='BAN',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/clip_vit-base-patch16-224_3rdparty-d08f8887.pth',
    asymetric_input=True,
    encoder_resolution=dict(
        size=(224, 224),
        mode='bilinear'),
    image_encoder=dict(
        type='mmseg.VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        patch_pad=0,
        in_channels=9,  # Changed from 3 to 9 for multi-channel input
        embed_dims=768,
        num_layers=9,
        num_heads=12,
        mlp_ratio=4,
        out_origin=False,
        out_indices=(2, 5, 8),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=True,
        patch_bias=False,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        act_cfg=dict(type='mmseg.QuickGELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        frozen_exclude=['pos_embed']),
    decode_head=dict(
        type='BitemporalAdapterHead',
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ban_cfg=dict(
            clip_channels=768,
            fusion_index=[1, 2, 3],
            side_enc_cfg=dict(
                type='mmseg.MixVisionTransformer',
                init_cfg=dict(
                    type='Pretrained', checkpoint=checkpoint_side),
                in_channels=9,  # Changed from 3 to 9
                embed_dims=32,
                num_stages=4,
                num_layers=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1)),
        ban_dec_cfg=dict(
            type='BAN_MLPDecoder',
            in_channels=[32, 64, 160, 256],
            channels=128,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============================================================================
# Training Configuration
# ============================================================================

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.),
            'mask_decoder': dict(lr_mult=10.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

# Learning rate scheduler
# Dataset: 1,028 train samples, batch_size=4 -> ~257 iters/epoch
# Total: 100 epochs = ~25,700 iterations
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=500,
        end=25000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# Training loop
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=25000,  # ~100 epochs
    val_interval=2500)  # Validate every ~10 epochs

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='CDVisualizationHook',
        interval=1,
        img_shape=(256, 256, 9)))  # 9 channels

# Runtime
work_dir = './experiments/ban'
