# 测试 Multi-Scale Deformable Attention 是否可用
import torch
from models.ops.functions import MSDeformAttnFunction  # 路径按你项目调整

value = torch.rand(1, 2, 128, 256).cuda()  # (bs, num_levels, num_points, embed_dim)
spatial_shapes = torch.tensor([[64, 64], [32, 32]]).cuda()
level_start_index = torch.tensor([0, 64*64]).cuda()
sampling_locations = torch.rand(1, 128, 4, 2, 2).cuda()
attention_weights = torch.rand(1, 128, 4, 2, 2).cuda()

out = MSDeformAttnFunction.apply(
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 64
)
print("✅ MSDA works!", out.shape)