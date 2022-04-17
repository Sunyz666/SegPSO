import torch
import torch.nn as nn
import scipy.ndimage as nd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.7, min_kept=80000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        print('factior=', self.factor)
        
    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        '''
        nd.zoom:
        Bilinear interpolation would be order=1, 
        nearest is order=0, 
        and cubic is the default (order=3).
        '''
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1) 
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor)  

        input_label = target.ravel().astype(np.int32) 
        input_prob = np.rollaxis(predict, 1).reshape((c, -1)) 

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept)-1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            # print(pred ,threshold)
            # print(pred <= threshold)
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            #print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target, len(valid_inds), threshold

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target, labels, thresh  = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target), labels, thresh
    
    
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, min_kept=80000, thresh=0.7, factor=8, ignore_index=255, cuda=True):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.ohem = OhemCrossEntropy2d(ignore_label=255, thresh=thresh, min_kept=min_kept, factor=factor)
        
    def build_loss(self, mode='ce'):
        """Choices: ['ce','focal', 'ce_ohem']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'ce_ohem':
            return self.ohem
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        if self.size_average:
            criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction = 'elementwise_mean')
        else:
            criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss




class diceCoeff(nn.Module):

    def __init__(self,activation = "sigmoid"):
        super(diceCoeff,self).__init__()
        self.activation =activation

    def forward(self,pred, gt, smooth=1e-5):
        if self.activation is None or self.activation == "none":
                activation_fn = lambda x: x
        elif self.activation == "sigmoid":
                activation_fn = nn.Sigmoid()
        elif self.activation == "softmax2d":
                activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
    
        pred = activation_fn(pred)
    
        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
    
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        loss = (2 * intersection + smooth) / (unionset + smooth)
        
        return loss.sum()/N

