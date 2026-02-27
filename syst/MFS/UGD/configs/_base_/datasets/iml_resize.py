# dataset settings
dataset_type = 'IMLDataset'
data_root = '/home/hexingyang/MMFusion-IML-main'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    
    
    dict(type='SizeResize',  size=(512,512)),
    dict(type='JPEGCompression', quality_lower=30, quality_upper=100, p=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    
    
    
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='JPEGCompression', quality_lower=50, quality_upper=51, p=1),
            dict(type='Guassblur',kernel_size=13),
            # dict(type='Guassnoise',var=13),
            # dict(type='gamma_correction',gamma=0.7),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
val_dataroot='/home/hexingyang/DDP/segmentation/data/IML'
test_dataroot='/home/disk/hexingyang/imltest/Casiav1'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='data',
        ann_dir='data',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=val_dataroot,
        img_dir='images',
        ann_dir='annotations',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=test_dataroot,
        img_dir='img',
        img_suffix='jpg',
        ann_dir='gt',
        seg_map_suffix='png',
        pipeline=test_pipeline))
