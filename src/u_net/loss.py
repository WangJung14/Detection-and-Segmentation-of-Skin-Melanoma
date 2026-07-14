import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss cho bài toán Image Segmentation.

    Công thức:
        Dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)

    Loss = 1 - Dice
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        # BCEWithLogits -> output chưa qua Sigmoid
        predictions = torch.sigmoid(predictions)

        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (predictions * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            predictions.sum() +
            targets.sum() +
            self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss

    Loss = BCE + Dice

    Đây là Loss được sử dụng rất phổ biến trong
    các bài toán Medical Image Segmentation.
    """

    def __init__(self,
                 bce_weight=0.5,
                 dice_weight=0.5):

        super(BCEDiceLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):

        bce_loss = self.bce(predictions, targets)

        dice_loss = self.dice(predictions, targets)

        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss
        )

        return total_loss