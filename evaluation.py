import numpy as np
from medpy.metric import dc, hd95, assd

def compute_binary_dice(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0:
        return 1 if num_pred == 0 else 0
    return dc(pred, gt)

def compute_binary_mIOU(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    return (np.sum(intersection) + 1e-5) / (np.sum(union) + 1e-5)

def compute_binary_HD95(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0 and num_pred == 0:
        return 0
    if num_gt == 0 or num_pred == 0:
        return 373.12866
    return hd95(pred, gt, (1, 1))

def compute_binary_ASSD(gt, pred):
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)
    if num_gt == 0 and num_pred == 0:
        return 0
    if num_gt == 0 or num_pred == 0:
        return 373.12866
    return assd(pred, gt, (1, 1))

def compute_multi_dice(gt, pred):
    return [compute_binary_dice(gt[i], pred[i]) for i in range(gt.shape[0])]

def compute_multi_mIOU(gt, pred):
    return [compute_binary_mIOU(gt[i], pred[i]) for i in range(gt.shape[0])]

def compute_multi_HD95(gt, pred):
    return [compute_binary_HD95(gt[i], pred[i]) for i in range(gt.shape[0])]

def compute_multi_ASSD(gt, pred):
    return [compute_binary_ASSD(gt[i], pred[i]) for i in range(gt.shape[0])]

def compute_seg_metrics(gt, pred):
    result = {}
    gt   = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    result['Dice'] = compute_binary_dice(gt, pred)
    result['IoU']  = compute_binary_mIOU(gt, pred)
    result['HD95'] = compute_binary_HD95(gt, pred)
    result['ASSD'] = compute_binary_ASSD(gt, pred)
    return result