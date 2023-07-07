import torch.nn as nn
import torch.nn.functional as F

from .bce_loss import BCELoss
from .dice_loss import DiceLoss

class SegLoss(nn.Module):
    def __init__(self, loss_func='dice', activation='sigmoid'):
        super(SegLoss, self).__init__()
        assert loss_func in {'dice', 'diceAndBce'}
        assert activation in {'sigmoid', 'softmax'}
        self.loss_func = loss_func
        self.activation = activation

    def forward(self, predict, gt, is_average=True):
        predict = predict.float()
        gt = gt.float()
        if self.activation == 'softmax':
            predict = F.softmax(predict, dim=1)
        elif self.activation == 'sigmoid':
            predict = F.sigmoid(predict)

        dice_loss_func = DiceLoss()
        loss = dice_loss_func(predict, gt, is_average)

        if self.loss_func == 'diceAndBce':
            bce_loss_func = BCELoss()
            loss += bce_loss_func(predict, gt, is_average)

        return loss
