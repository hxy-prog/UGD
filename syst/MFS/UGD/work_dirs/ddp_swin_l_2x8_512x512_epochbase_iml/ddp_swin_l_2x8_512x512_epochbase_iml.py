dataset_type = 'IMLDataset'
data_root = '/home/hexingyang/MMFusion-IML-main'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', ratio_range=(0.5, 1.5), p=0.5),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='JPEGCompression', quality_lower=30, quality_upper=100, p=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
val_dataroot = '/home/hexingyang/DDP/segmentation/data/IML'
test_dataroot = '/home/disk/hexingyang/imltest/DSO-1'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='IMLDataset',
        data_root='/home/hexingyang/MMFusion-IML-main',
        img_dir='data',
        ann_dir='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', ratio_range=(0.5, 1.5), p=0.5),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
            dict(type='RandomCrop', crop_size=(512, 512)),
            dict(
                type='JPEGCompression',
                quality_lower=30,
                quality_upper=100,
                p=0.5),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='IMLDataset',
        data_root='/home/hexingyang/DDP/segmentation/data/IML',
        img_dir='images',
        ann_dir='annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='IMLDataset',
        data_root='/home/disk/hexingyang/imltest/DSO-1',
        img_dir='img',
        img_suffix='png',
        ann_dir='gt',
        seg_map_suffix='png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DDP',
    timesteps=1000,
    bit_scale=0.01,
    accumulation=True,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
        ),
        pretrain_img_size=384,
        in_channels=3,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True)),
    neck=[
        dict(
            type='FPN',
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        dict(
            type='MultiStageMerging',
            in_channels=[256, 256, 256, 256],
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=None)
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    decode_head=dict(
        type='DeformableHeadWithTime',
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.0,
        num_classes=2,
        out_channels=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.0),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.0,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                class_weight=[0.5, 2.5],
                loss_weight=0.3,
                ignore_index=-1),
            dict(
                type='DiceLoss',
                class_weight=[0.5, 2.5],
                loss_weight=0.7,
                ignore_index=-1)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = './work_dirs/ddp_swin_l_2x8_512x512_epochbase_iml'
gpu_ids = range(0, 1)
auto_resume = False
