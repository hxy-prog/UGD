import torch
from torch import Tensor


class DiceLoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.eps = 1e-9
        self.ignore_index = ignore_index
        
    def forward(self, logits: Tensor, target: Tensor):
        bs, c, h, w = logits.size()
        if bs == 0:
            return torch.tensor(0.).to(logits.device)

        pred = torch.softmax(logits, dim=1)[:, 1, :, :]

        not_ignored_mask = target != self.ignore_index
        true = target * not_ignored_mask
        pred = pred * not_ignored_mask

        intersection = (pred * true).sum(dim=(-1, -2))
        union = (pred + true).sum(dim=(-1, -2))

        dice_losses = 2. * intersection / (union + self.eps)
        dice_loss_batch = dice_losses.mean()

        return 1 - dice_loss_batch


class TruForLoss(torch.nn.Module):
    def __init__(self, lambda_ce: float = 0.3, ignore_index: int = -1, weights=torch.tensor([0.5, 2.5], device='cuda:0')):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.ignore_index = ignore_index
        self.criterion_bce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index)
        self.criterion_dice = DiceLoss(ignore_index=self.ignore_index)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:

        loss_bce = self.criterion_bce(logits, target)
        loss_dice = self.criterion_dice(logits, target)
        loss = self.lambda_ce * loss_bce + (1 - self.lambda_ce) * loss_dice
        return loss
##criterion = TruForLoss(weights=train.class_weights.to(device), ignore_index=-1)