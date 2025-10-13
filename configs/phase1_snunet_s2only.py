_base_ = [
    '../open-cd/configs/_base_/models/snunet_c16.py',
    '../open-cd/configs/_base_/default_runtime.py'
]

# Dataset
dataset_type = 'CustomCDDataset'
data_root = 'data/samples/phase1_s2only'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406] * 2,
    std=[0.229, 0.224, 0.225] * 2,
    to_rgb=False
)

crop_size = (256, 256)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgNormalize', **img_norm_cfg),
    dict(type='MultiImgDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgMultiScaleFlipAug',
         img_scale=(256, 256),
         flip=False,
         transforms=[
             dict(type='MultiImgNormalize', **img_norm_cfg),
             dict(type='MultiImgImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img'])
         ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test',
        pipeline=test_pipeline))

model = dict(
    type='SiamEncoderDecoder',
    backbone=dict(
        type='SNUNet_ECAM',
        in_channels=7,  # S2 only: 4 bands + 3 indices
        width=16,
        enc_depth=[2, 2, 2, 2],
        enc_channels=[16, 32, 64, 128],
        dec_channels=[128, 64, 32, 16]
    ),
    decode_head=dict(
        type='SNUNetHead',
        in_channels=16,
        channels=16,
        num_classes=2,
        dropout_ratio=0.1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optimizer = dict(type='AdamW', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])

work_dir = 'experiments/phase1_s2only'
