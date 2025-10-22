"""
SNUNet-CD Configuration for Ca Mau Forest Change Detection
Author: Ninh Hai Dang (21021411)
Date: 2025-10-17

Model: SNUNet-CD with Dense Connections and Channel Attention (ECAM)
Input: 9 channels per time step (S2 + S1: B4, B8, B11, B12, NDVI, NBR, NDMI, VH, Ratio)
Dataset: Ca Mau forest change (1,285 samples: 1,028 train, 128 val, 129 test)
"""

# Default runtime settings
default_scope = 'opencd'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=640, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, img_shape=(256, 256, 9)))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='CDLocalVisBackend')]
visualizer = dict(
    type='CDLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    alpha=1.0)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='mmseg.SegTTAModel')

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
    batch_size=16,  # OPTIMIZED: Balance speed vs stability
    num_workers=2,  # OPTIMIZED: Reduce RAM overhead
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

base_channels = 32  # SNUNet with 32 base channels for better capacity
model = dict(
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # SNUNet-CD trains from scratch
    backbone_inchannels=9,  # IMPORTANT: Set to 9 for 9-channel input per timestep
    backbone=dict(
        type='SNUNet_ECAM',
        in_channels=9,  # Changed from 3 to 9 for multi-channel input
        base_channel=base_channels),
    decode_head=dict(
        type='mmseg.FCNHead',
        in_channels=base_channels * 4,
        channels=base_channels * 4,
        in_index=-1,
        num_convs=0,
        concat_input=False,
        num_classes=2,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ============================================================================
# Training Configuration
# ============================================================================

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer)

# Learning rate scheduler
# Dataset: 1,028 train samples, batch_size=16 -> ~64 iters/epoch
# Total: 100 epochs = ~6,400 iterations
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=256),
    dict(
        type='PolyLR',
        power=1.0,
        begin=256,
        end=6400,
        eta_min=0.0,
        by_epoch=False,
    )
]

# Training loop
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=6400,  # ~100 epochs with batch_size=16
    val_interval=640)  # Validate every ~10 epochs

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
        interval=640,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='CDVisualizationHook',
        interval=1,
        img_shape=(256, 256, 9)))  # 9 channels

# Runtime
work_dir = './experiments/snunet'
