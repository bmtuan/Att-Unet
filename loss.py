import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(outputs, labels, reduction="mean"):
    # mask for labeled pixel
    # loss    = F.cross_entropy(outputs.float(), torch.argmax(labels, dim=1))
    loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.tensor([0.6]).cuda())
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("reduction must be either mean or sum")


def new_loss(outputs, labels):
    logpt = -F.binary_cross_entropy_with_logits(outputs, labels, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt) ** 2) * logpt
    return loss.mean()


def iou_loss(outputs, labels, reduction="mean"):
    smooth = 1e-6
    b, c, h, w = outputs.size()
    outputs = F.sigmoid(outputs)
    loss = []
    for i in range(c):
        o_flat = outputs[:, i, :, :].contiguous().view(-1)
        l_flat = labels[:, i, :, :].contiguous().view(-1)
        intersection = torch.sum(o_flat * l_flat)
        total = torch.sum(o_flat + l_flat)
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        loss.append(1 - iou)
    loss = torch.stack(loss)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("reduction must be either mean or sum")


def dice_loss(outputs, labels, reduction="mean"):
    smooth = 1e-7

    b, c, h, w = outputs.size()
    loss = []
    for i in range(c):
        o_flat = outputs[:, i, :, :].contiguous().view(-1)
        l_flat = labels[:, i, :, :].contiguous().view(-1)
        intersection = torch.sum(o_flat * l_flat)
        union = torch.sum(o_flat * o_flat) + torch.sum(l_flat * l_flat)

        loss.append(1 - (2. * intersection + smooth) / (union + smooth))

    loss = torch.stack(loss)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("reduction must be either mean or sum")


def dbce_loss(outputs, labels, reduction="mean"):
    """Dice loss + BCE loss"""
    # mask for labeled pixel
    bce_loss2    = bce_loss(outputs, labels, reduction=reduction)
    dice_loss2   = dice_loss(outputs, labels, reduction=reduction)
    iou_loss2    = iou_loss(outputs, labels, reduction=reduction)
    # print(f'bce_loss: {bce_loss2}, dice_loss: {dice_loss2}')
    return (bce_loss2 + dice_loss2 + iou_loss2) / 2.0
