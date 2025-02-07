import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, average_precision_score

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    # print(true.get_device())
    # print(logits.get_device())
    # print('logits type: ', logits.dtype)
    # print(logits)
    
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        # print(num_classes, ' ',  true.shape)
        # print('eye shape: ', torch.eye(num_classes).shape)
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        # print('after---: ', true_1_hot.shape)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # print('one hot gt shape: ', true_1_hot.shape)
        # print('predicted shape: ', logits.shape)
        probas = F.softmax(logits, dim=1)
        # print('probabs: ', torch.max(probas), probas)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    # print('uniques: ', torch.unique(probas), torch.unique(true_1_hot))
    intersection = torch.sum(probas * true_1_hot, dims)
    # print('intersection: ', intersection)
    cardinality = torch.sum(probas + true_1_hot, dims)
    # print('union: ', cardinality)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def dice_loss_2d(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    # print(true.get_device())
    # print(logits.get_device())
    # print('logits type: ', logits.dtype)
    # print(logits)
    
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        # print(num_classes, ' ',  true.shape)
        # print('eye shape: ', torch.eye(num_classes).shape)
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        # print('after---: ', true_1_hot.shape)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        probas = F.softmax(logits, dim=1)
        # print('probabs: ', torch.max(probas), probas)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    # print('uniques: ', torch.unique(probas), torch.unique(true_1_hot))
    intersection = torch.sum(probas * true_1_hot, dims)
    # print('intersection: ', intersection)
    cardinality = torch.sum(probas + true_1_hot, dims)
    # print('union: ', cardinality)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss_2d(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def dice_coeff(pred, target):
    target = target.contiguous()
    # print('pred and target shapes: ', pred.shape, ' ', target.shape)
    smooth = 0.001
    #print(pred.shape, pred.shape[0])
    #print('--size: ', torch.Tensor(pred).size(0))
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    #print('reshaped shapes: ', m1.shape, ' ', m2.shape)
    intersection = (m1 * m2).sum()
    
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    iou = (intersection + smooth) / (m1.sum() + m2.sum() - intersection + smooth)
    return dice, iou

def weights(pred, target, epsilon = 1e-6):
    num_classes = 4
    pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = np.ones(num_classes)
    tot = 0
    for c in range(num_classes):
        t = (target == c).sum()
        tot = tot + t
        #print(t.shape)
        dice[c] = t

    dice = dice/dice.sum()
    dice = 1 - dice
    
    return torch.from_numpy(dice).float()

def class_dice(pred_class, target, tot_classes, epsilon = 1e-6):
    num_classes = torch.unique(target)
    #print('num classes: ', num_classes)
    #pred_class = torch.argmax(pred, dim = 1)
    #pred_class = torch.argmax(pred.squeeze(), dim=1).detach().cpu().numpy()
    dice = torch.zeros(3)
    dscore = torch.zeros(3)
    iou_score = torch.zeros(3)
    ds = 0
    ious = 0
    for c in range(1, tot_classes):
        if c in num_classes:
            #print('c: ', c)
            p = (pred_class == c)
            t = (target == c)

            dc, iou = dice_coeff(p, t)
            #print('dc done')
            dice[c-1] = 1 - dc
            dscore[c-1] = dc
            iou_score[c-1] = iou
            #print('appended')
            dl = torch.sum(dice)
            ds = torch.mean(dscore)
            ious = torch.mean(iou_score)
        
    return ds, dscore, ious, iou_score

def loss_function(pred, gt):
    ce_loss = nn.CrossEntropyLoss()#weight = weights(pred, gt).cuda())

    dsc_l = dice_loss(gt, pred)
    ce_l = ce_loss(pred, gt)

    loss = ce_l*0.5 + dsc_l*0.5

    return loss