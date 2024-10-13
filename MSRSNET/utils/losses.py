import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        # batch equal to True means views all batch images as an entity and calculate loss
        # batch equal to False means calculate loss of every single image in batch and get their mean
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_bce_loss(nn.Module):

    def __init__(self):
        super(dice_bce_loss, self).__init__()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels)
        foclaloss = self.focal_loss(scores.clone(), labels)
        return [diceloss, foclaloss]

class dice_Focal_loss(nn.Module):
    def __init__(self):
        super(dice_Focal_loss, self).__init__()
        self.focal_loss = FocalLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels)
        foclaloss = self.focal_loss(scores.clone(), labels)
        return [diceloss, foclaloss]

def FCCDN_loss_without_seg(scores, labels):
    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
    """ for binary change detection task"""
    criterion_change = dice_bce_loss()

    # change loss
    loss_change = criterion_change(scores[0], labels[0])
    loss_seg1 = criterion_change(scores[1], labels[1])
    loss_seg2 = criterion_change(scores[2], labels[1])

    for i in range(len(loss_change)):
        loss_change[i] += 0.2 * (loss_seg1[i] + loss_seg2[i])

    return loss_change

def FSCDN_Loss_without_seg(scores, labels):
    scores = [score.squeeze(1) if len(score.shape) > 3 else score for score in scores]
    labels = [label.squeeze(1) if len(label.shape) > 3 else label for label in labels]
    """ for binary change detection task"""
    criterion_change = dice_Focal_loss()

    # change loss
    loss_change = criterion_change(scores[0], labels[0])
    loss_seg1 = criterion_change(scores[1], labels[1])
    loss_seg2 = criterion_change(scores[2], labels[1])

    for i in range(len(loss_change)):
        loss_change[i] += 0.2 * (loss_seg1[i] + loss_seg2[i])

    return loss_change


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ShapeAwareLoss(nn.Module):
#     def __init__(self, weight=1.0):
#         super(ShapeAwareLoss, self).__init__()
#         self.weight = weight
#
#     def forward(self, prediction, target):
#         # Compute binary edge maps from target masks
#         target_edge = self.compute_edge_map(target)
#
#         # Compute edge map from predicted masks
#         pred_edge = self.compute_edge_map(prediction)
#
#         # Compute binary cross entropy loss for edge maps
#         bce_loss = F.binary_cross_entropy(pred_edge, target_edge)
#
#         return bce_loss * self.weight
#
#     def compute_edge_map(self, mask):
#         # Use Sobel filter to compute gradient magnitude
#         dx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         dy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#
#         gradient_x = F.conv2d(mask, dx)
#         gradient_y = F.conv2d(mask, dy)
#
#         gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
#
#         # Apply thresholding to get binary edge map
#         edge_map = torch.where(gradient_magnitude > 0.5, torch.tensor(1), torch.tensor(0))
#
#         return edge_map
