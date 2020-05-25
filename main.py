import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset.data import VideoTextDataset
from model.acga import ACGA
from util.tool import AverageMeter, get_video_spatial_feature, save_checkpoint, adjust_learning_rate, report_result


def prepare_environment(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(opt.seed)
    print(f'environment prepared done: {opt}')


def prepare_model(opt):
    model = ACGA(opt)
    model = nn.DataParallel(model).cuda()

    param_list = [
        {'params': model.module.video_model.model.mixed_4f.parameters()},
        {'params': model.module.txt_model.parameters()},
        {'params': model.module.attention_layer.parameters()},
        {'params': model.module.deconv_small.parameters()},
        {'params': model.module.conv_mask_small.parameters()},
        {'params': model.module.deconv_medium.parameters()},
        {'params': model.module.conv_mask_medium.parameters()},
        {'params': model.module.conv_mask_large.parameters()},
        {'params': model.module.conv_fusion.parameters()}
    ]
    optimizer = optim.Adam(param_list, lr=opt.lr)

    return model, optimizer


def train(opt, epoch, dataset, model, optimizer, spatial_feature_small_org, spatial_feature_medium_org, spatial_feature_large_org):
    losses_res1 = AverageMeter()
    losses_res2 = AverageMeter()
    losses_res3 = AverageMeter()

    model.train()

    lr = adjust_learning_rate(opt, epoch, optimizer)

    for _ in tqdm(range(dataset.ndata//opt.batch_size)):
        size, video, txt, mask_small, mask_medium, mask_large, bbox_small, bbox_medium, bbox_large = dataset.next_batch(opt.batch_size)
        video, txt, mask_small, mask_medium, mask_large = video.cuda(), txt.cuda(), mask_small.cuda(), mask_medium.cuda(), mask_large.cuda()

        spatial_feature_small = torch.from_numpy(spatial_feature_small_org).unsqueeze(0).repeat(video.size(0), 1, 1, 1).cuda()
        spatial_feature_medium = torch.from_numpy(spatial_feature_medium_org).unsqueeze(0).repeat(video.size(0), 1, 1, 1).cuda()
        spatial_feature_large = torch.from_numpy(spatial_feature_large_org).unsqueeze(0).repeat(video.size(0), 1, 1, 1).cuda()

        pred_res1, pred_res2, pred_res3 = model(video, txt, spatial_feature_small, spatial_feature_medium, spatial_feature_large)
        loss_res1 = F.binary_cross_entropy_with_logits(pred_res1, mask_small, pos_weight=torch.tensor(opt.pos_weight))
        loss_res2 = F.binary_cross_entropy_with_logits(pred_res2, mask_medium, pos_weight=torch.tensor(opt.pos_weight))
        loss_res3 = F.binary_cross_entropy_with_logits(pred_res3, mask_large, pos_weight=torch.tensor(opt.pos_weight))
        loss = loss_res1 + loss_res2 + loss_res3

        losses_res1.update(loss_res1.item(), video.size(0))
        losses_res2.update(loss_res2.item(), video.size(0))
        losses_res3.update(loss_res3.item(), video.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, LR {lr:.7f}, LossRes1 {losses_res1.avg:.5f}, LossRes2 {losses_res2.avg:.5f}, LossRes3 {losses_res3.avg:.5f}')


def test(opt, savedir, dataset, model, spatial_feature_small_org, spatial_feature_medium_org, spatial_feature_large_org):
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    resume = os.path.join(opt.project_root, 'checkpoint', savedir, 'model_best.pth.tar')

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        opt.start_epoch = checkpoint['epoch']

        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        if len(trash_vars) > 0:
            print(f'trashed vars from resume dict: {trash_vars}')
        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    model.eval()
    mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP = \
        report_result(opt, dataloader, model, spatial_feature_small_org, spatial_feature_medium_org, spatial_feature_large_org)
    print(f'Test split results:\n'
          f'Precision@0.5 {precision5:.3f}, Precision@0.6 {precision6:.3f}, '
          f'Precision@0.7 {precision7:.3f}, Precision@0.8 {precision8:.3f}, Precision@0.9 {precision9:.3f},\n'
          f'mAP Precision @0.5:0.05:0.95 {precision_mAP:.3f},\n'
          f'Overall IoU {overall_iou:.3f}, Mean IoU {mean_iou:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment on Actor and Action Video Segmentation')
    # Project Structure
    parser.add_argument('--project_root', type=str, default='/home/user/ACGASegWithLang/')
    parser.add_argument('--savedir_root', type=str, default='/home/user/ACGASegWithLang/')
    # Dataset Specific
    parser.add_argument('--dataset', type=str, choices=['A2D', 'JHMDB'])
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'flow'])
    parser.add_argument('--version', type=str, default='iccv', choices=['iccv'])
    # Model Specific
    parser.add_argument('--arch', type=str, default='ACGA', choices=['ACGA'])
    parser.add_argument('--frame_length', type=int, default=16)
    parser.add_argument('--image_large_size', type=int, default=512)
    parser.add_argument('--image_medium_size', type=int, default=128)
    parser.add_argument('--image_small_size', type=int, default=32)
    parser.add_argument('--sentence_length', type=int, default=20)
    parser.add_argument('--dim_semantic', type=int, default=300)
    parser.add_argument('--dim_spatial', type=int, default=8)
    parser.add_argument('--dim_visual_small', type=int, default=832)
    parser.add_argument('--dim_visual_medium', type=int, default=256)
    parser.add_argument('--dim_visual_large', type=int, default=128)
    # Traing Specific
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--nepoch', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr_step', type=int, default=8)
    parser.add_argument('--gamma_step', type=float, default=0.1)
    parser.add_argument('--seed', default=2019)
    parser.add_argument('--gpu_id', type=int, default=0)
    # Misc
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--pos_weight', type=float, default=1.5)
    opt = parser.parse_args()

    prepare_environment(opt)
    model, optimizer = prepare_model(opt)
    dataset = VideoTextDataset(opt)

    spatial_feature_small = get_video_spatial_feature(opt.image_small_size, opt.image_small_size)
    spatial_feature_medium = get_video_spatial_feature(opt.image_medium_size, opt.image_medium_size)
    spatial_feature_large = get_video_spatial_feature(opt.image_large_size, opt.image_large_size)

    savedir = f'{opt.arch}'
    if not os.path.exists(f'{opt.project_root}/checkpoint/{savedir}'):
        os.makedirs(f'{opt.project_root}/checkpoint/{savedir}')

    if not opt.testing:
        for epoch in range(opt.nepoch):
            train(opt, epoch, dataset, model, optimizer, spatial_feature_small, spatial_feature_medium, spatial_feature_large)

            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True, filename=f'{opt.project_root}/checkpoint/{savedir}/checkpoint.pth.tar')
    else:
        test(opt, savedir, dataset, model, spatial_feature_small, spatial_feature_medium, spatial_feature_large)
