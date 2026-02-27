# optimizer 配置保持不变
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy - 注意这里假设总的训练epoch数量是已知的，例如假设为200 epochs
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True, max_epochs=50)

# runtime settings - 改为基于Epoch的Runner，并设置合适的epoch数
runner = dict(type='EpochBasedRunner', max_epochs=50)

# checkpoint配置 - 每20个epoch保存一次
checkpoint_config = dict(by_epoch=True, interval=5)

# evaluation配置 - 每个epoch结束时进行验证
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
