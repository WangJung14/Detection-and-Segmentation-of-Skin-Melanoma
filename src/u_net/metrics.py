import torch


def dice_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Dice Score

    Giá trị:
        0 -> rất tệ
        1 -> hoàn hảo
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum() +
        targets.sum() +
        smooth
    )

    return dice.item()


def iou_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Intersection over Union (IoU)

    Giá trị:
        0 -> rất tệ
        1 -> hoàn hảo
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    union = (
        predictions.sum() +
        targets.sum() -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()


def pixel_accuracy(predictions, targets, threshold=0.5):
    """
    Pixel Accuracy

    Tỷ lệ số pixel dự đoán đúng.
    """

    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()

    correct = (predictions == targets).float().sum()

    total = torch.numel(predictions)

    accuracy = correct / total

    return accuracy.item()


def evaluate_batch(predictions, targets):
    """
    Trả về tất cả metrics của một batch.
    """

    return {
        "dice": dice_score(predictions, targets),
        "iou": iou_score(predictions, targets),
        "pixel_acc": pixel_accuracy(predictions, targets)
    }