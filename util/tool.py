import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        filepath = '/'.join(filename.split('/')[0:-1])
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def adjust_learning_rate(opt, epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (opt.gamma_step ** (epoch // opt.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_video_spatial_feature(featmap_H, featmap_W):
    spatial_batch_val = np.zeros((8, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[:, h, w] = [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
    return spatial_batch_val


def resize_and_crop(im, input_h, input_w):
    # Resize and crop im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = max(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    resized_im = cv2.resize(im, (resized_w, resized_h))
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[...] = resized_im[crop_h:crop_h+input_h, crop_w:crop_w+input_w, ...]
    return new_im


SMOOTH = 1e-6
def calculate_IoU(pred, gt):
    IArea = (pred & (gt == 1.0)).astype(float).sum()
    OArea = (pred | (gt == 1.0)).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    return IoU, IArea, OArea


def report_result(opt, dataloader, model, spatial_feature_small_org, spatial_feature_medium_org, spatial_feature_large_org):
    MeanIoU, IArea, OArea, Overlap = [], [], [], []

    for size, video, txt, mask_small, mask_medium, mask_large, bbox_small, bbox_medium, bbox_large in tqdm(dataloader):
        video, txt = video.cuda(), txt.cuda()
        size, mask_large = size.numpy(), mask_large.numpy()

        spatial_feature_small = torch.from_numpy(spatial_feature_small_org).unsqueeze(0).repeat(video.size(0), 1, 1, 1).cuda()
        spatial_feature_medium = torch.from_numpy(spatial_feature_medium_org).unsqueeze(0).repeat(video.size(0), 1, 1, 1).cuda()
        spatial_feature_large = torch.from_numpy(spatial_feature_large_org).unsqueeze(0).repeat(video.size(0), 1, 1, 1).cuda()

        with torch.no_grad():
            _, _, pred_res3 = model(video, txt, spatial_feature_small, spatial_feature_medium, spatial_feature_large)

            res3 = torch.sigmoid(pred_res3) * 255.0
            res3 = res3.detach().cpu().numpy()
            pred = [resize_and_crop((res3[i] > np.amax(res3[i]) * 0.5).astype(np.uint8), size[i][0], size[i][1]) for i in range(res3.shape[0])]
            gt = [resize_and_crop((mask_large[i]).astype(np.uint8), size[i][0], size[i][1]) for i in range(res3.shape[0])]

            for i in range(len(pred)):
                iou, iarea, oarea = calculate_IoU(pred[i], gt[i])
                MeanIoU.append(iou)
                IArea.append(iarea)
                OArea.append(oarea)
                Overlap.append(iou)

    prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), \
                                        np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
    for i in range(len(Overlap)):
        if Overlap[i] >= 0.5:
            prec5[i] = 1
        if Overlap[i] >= 0.6:
            prec6[i] = 1
        if Overlap[i] >= 0.7:
            prec7[i] = 1
        if Overlap[i] >= 0.8:
            prec8[i] = 1
        if Overlap[i] >= 0.9:
            prec9[i] = 1

    # maybe different with coco style as we could not get detailed response about the way to calculate it.
    # it is confuse to define precision and recall for me, if we follow the prior definition of precision.
    # anyone is welcome to pull request
    mAP_thres_list = list(range(50, 95+1, 5))
    mAP = []
    for i in range(len(mAP_thres_list)):
        tmp = np.zeros((len(Overlap), 1))
        for j in range(len(Overlap)):
            if Overlap[j] >= mAP_thres_list[i] / 100.0:
                tmp[j] = 1
        mAP.append(tmp.sum() / tmp.shape[0])

    return np.mean(np.array(MeanIoU)), np.array(IArea).sum() / np.array(OArea).sum(), \
           prec5.sum() / prec5.shape[0], prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], \
           prec8.sum() / prec8.shape[0], prec9.sum() / prec9.shape[0], np.mean(np.array(mAP))
