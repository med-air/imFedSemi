import torch
import torch.nn
from torch.nn import functional as F
import numpy as np

"""
The different loss implementation.
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [141, 927, 679, 1125, 2136]
CLASS_WEIGHT = torch.Tensor([5000/i for i in CLASS_NUM]).cuda()



def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(object):
    def __init__(self, epsilon = 0.1, reduction='mean'):
        #super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.base_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=CLASS_WEIGHT)

    def __call__(self, preds, target):
        target = torch.argmax(target, dim=1)
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target.long(), reduction=self.reduction)
        #output_softmax = F.softmax(preds, dim=1)
        #target = torch.argmax(target, dim=1)
        #nll = self.base_loss(output_softmax, target.long())
        return linear_combination(loss / n, nll, self.epsilon)

class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=CLASS_WEIGHT)
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())


def get_UncertaintyLoss(method):
    assert method in METHODS
    
    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()
    
    if method == 'U-MultiClass':
        return Loss_MultiClass()

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent



def entropy_loss(p, c=2):
    # p N*C*W*H*D
     p = F.softmax(p, dim=1)
     y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()
     ent = torch.mean(y1)
     return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2 
    return mse_loss


