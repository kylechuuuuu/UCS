import torch
import numpy as np
import cv2
import pdb
import math
def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()


def f1(pr, gt, epsilon=1e-7, threshold = 0.5):
    # 1, h, w
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    tp = torch.sum(gt_ * pr_,dim=[1,2,3]).to(torch.float32)
    tn = torch.sum((1 - gt_) * (1 - pr_),dim=[1,2,3]).to(torch.float32)
    fp = torch.sum((1 - gt_) * pr_,dim=[1,2,3]).to(torch.float32)
    fn = torch.sum(gt_ * (1 - pr_),dim=[1,2,3]).to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + fp + tn + fn)
    aver_acc = (tp / (tp + fn) + tn / (tn + fp)) / 2
    mcc = (tp * tn - tp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    miou = tp / (tp + fp + fn)

    return f1_score.cpu().numpy()


def pre(pr, gt, epsilon=1e-7, threshold = 0.5):
    # 1, h, w
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    tp = torch.sum(gt_ * pr_,dim=[1,2,3]).to(torch.float32)
    tn = torch.sum((1 - gt_) * (1 - pr_),dim=[1,2,3]).to(torch.float32)
    fp = torch.sum((1 - gt_) * pr_,dim=[1,2,3]).to(torch.float32)
    fn = torch.sum(gt_ * (1 - pr_),dim=[1,2,3]).to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    return precision.cpu().numpy()


def rec(pr, gt, epsilon=1e-7, threshold = 0.5):
    # 1, h, w
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    tp = torch.sum(gt_ * pr_,dim=[1,2,3]).to(torch.float32)
    tn = torch.sum((1 - gt_) * (1 - pr_),dim=[1,2,3]).to(torch.float32)
    fp = torch.sum((1 - gt_) * pr_,dim=[1,2,3]).to(torch.float32)
    fn = torch.sum(gt_ * (1 - pr_),dim=[1,2,3]).to(torch.float32)
    recall = tp / (tp + fn + epsilon)
    return recall.cpu().numpy()


def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
        
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))

        elif metric == 'f1':
            metric_list.append(np.mean(f1(pred, label)))
        elif metric == 'pre':
            metric_list.append(np.mean(pre(pred, label)))
        elif metric == 'rec':
            metric_list.append(np.mean(rec(pred, label)))
            #pdb.set_trace()    
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric