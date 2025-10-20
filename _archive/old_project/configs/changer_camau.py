"""
Changer Configuration for Ca Mau Forest Change Detection
Author: Ninh Hai Dang (21021411)
Date: 2025-10-17

Model: Changer with MiT-B0 backbone (Interaction-Aware MixVisionTransformer)
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

# Training pipeline (with temporal exchange augmentation)
train_pipeline = [
    dict(type='MultiImgLoadRasterioFromFile'),  # Custom loader for 9-channel TIFF
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgExchangeTime', prob=0.5),  # Changer-specific augmentation
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
    batch_size=12,  # INCREASED: Moderate model, good GPU utilization
    num_workers=8,  # INCREASED: Use more CPU cores
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
    num_workers=8,
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
    num_workers=8,
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

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

model = dict(
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone_inchannels=9,  # IMPORTANT: Set to 9 for 9-channel input per timestep
    backbone=dict(
        type='IA_MixVisionTransformer',
        in_channels=9,  # Changed from 3 to 9 for multi-channel input
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
        drop_path_rate=0.1,
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2))),
    decode_head=dict(
        type='Changer',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============================================================================
# Training Configuration
# ============================================================================

# Optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# Learning rate scheduler
# Dataset: 1,028 train samples, batch_size=12 -> ~86 iters/epoch
# Total: 100 epochs = ~8,600 iterations (2x faster!)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=300),
    dict(
        type='PolyLR',
        power=1.0,
        begin=300,
        end=8600,
        eta_min=0.0,
        by_epoch=False,
    )
]

# Training loop
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=8600,  # ~100 epochs with batch_size=12
    val_interval=860)  # Validate every ~10 epochs

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
        interval=860,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='CDVisualizationHook',
        interval=1,
        img_shape=(256, 256, 9)))  # 9 channels

# Runtime
work_dir = './experiments/changer'
