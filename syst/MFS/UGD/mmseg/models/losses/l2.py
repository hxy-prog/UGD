import torch.nn.functional as F
from ..builder import LOSSES
import torch.nn as nn
@LOSSES.register_module()
class L2loss(nn.Module):
    def __init__(self):
        super(L2loss,self).__init__()
        self._loss_name='L2loss'
    def forward(self, pred,target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        # print('aaa')
        # print(pred.size())
        # print(target.size())
        # return F.l1_loss(pred.float(), target.float(), reduction='mean')
        return F.mse_loss(pred.float(), target.float(), reduction="mean")
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name