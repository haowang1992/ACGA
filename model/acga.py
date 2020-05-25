import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from .pytorch_i3d import I3D


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


class VideoEncoder(nn.Module):
    def __init__(self, opt):
        super(VideoEncoder, self).__init__()
        self.opt = opt

        if self.opt.modality == 'rgb':
            self.model = I3D(num_classes=400, modality=self.opt.modality)
            self.model.load_state_dict(torch.load(f'{self.opt.project_root}/model/i3d_{self.opt.modality}.pth'))
        elif self.opt.modality == 'flow':
            self.model = I3D(num_classes=400, modality=self.opt.modality)
            self.model.load_state_dict(torch.load(f'{self.opt.project_root}/model/i3d_{self.opt.modality}.pth'))
        else:
            raise RuntimeError('Modality only support rgb or flow')

    def forward(self, x):
        return self.model.mixed_4f(x)


class AttentionLayer(nn.Module):
    def __init__(self, opt):
        super(AttentionLayer, self).__init__()
        self.opt = opt

        self.video_reduce = nn.Linear(in_features=self.opt.dim_visual_small+self.opt.dim_spatial, out_features=self.opt.dim_semantic)
        self.video_linearK = nn.Linear(in_features=self.opt.dim_visual_small+self.opt.dim_spatial, out_features=self.opt.dim_visual_small+self.opt.dim_spatial)
        self.video_linearQ = nn.Linear(in_features=self.opt.dim_visual_small+self.opt.dim_spatial, out_features=self.opt.dim_visual_small+self.opt.dim_spatial)
        self.video_linearV = nn.Linear(in_features=self.opt.dim_visual_small+self.opt.dim_spatial, out_features=self.opt.dim_visual_small+self.opt.dim_spatial)

        self.txt_pool = nn.MaxPool1d(kernel_size=self.opt.sentence_length)
        self.txt_increase = nn.Linear(in_features=self.opt.dim_semantic, out_features=self.opt.dim_visual_small+self.opt.dim_spatial)

    def forward(self, video, txt, spatial):
        # video (N, 832, 32, 32), spatial (N, 8, 32, 32), txt (N, 300, 20)
        # (N, 832+8, 32, 32) -> (N, 32, 32, 832+8)
        video_spatial_org = torch.cat((video, spatial), dim=1).permute(0, 2, 3, 1).contiguous()

        # (N, 32, 32, 832+8) -> (N*32*32, 832+8) -> (N*32*32, 300)
        video_spatial_reduce = self.video_reduce(video_spatial_org.view(-1, self.opt.dim_visual_small+self.opt.dim_spatial))
        # (N * 32 * 32, 300) -> (N, 32*32, 300)
        video_spatial_reduce = video_spatial_reduce.view(-1, self.opt.image_small_size**2, self.opt.dim_semantic)
        # (N, 32*32, 300) * (N, 300, 20) -> (N, 32*32, 20)
        txt_score = F.softmax(torch.bmm(video_spatial_reduce, txt) / np.power(self.opt.dim_semantic, 0.5), dim=-1)
        # (N, 32*32, 20) * (N, 20, 300) -> (N, 32*32, 300) -> (N, 32, 32, 300)
        weighted_txt = torch.bmm(txt_score, txt.permute(0, 2, 1).contiguous()).view(-1, self.opt.image_small_size, self.opt.image_small_size, self.opt.dim_semantic)
        # (N, 300, 32, 32)
        weighted_txt = weighted_txt.permute(0, 3, 1, 2).contiguous()

        # (N, 300, 20) -> (N, 300) -> (N, 832+8) -> (N, 32*32, 832+8)
        txt_repeat = self.txt_increase(self.txt_pool(txt).squeeze(-1)).unsqueeze(1).repeat(1, self.opt.image_small_size**2, 1)
        # (N, 32*32, 832+8) -> (N, 32*32, 832+8)
        video_key = self.video_linearK(video_spatial_org.view(-1, self.opt.dim_visual_small+self.opt.dim_spatial)).view(-1, self.opt.image_small_size**2, self.opt.dim_visual_small+self.opt.dim_spatial) * txt_repeat
        video_query = self.video_linearQ(video_spatial_org.view(-1, self.opt.dim_visual_small+self.opt.dim_spatial)).view(-1, self.opt.image_small_size**2, self.opt.dim_visual_small+self.opt.dim_spatial) * txt_repeat
        video_value = self.video_linearV(video_spatial_org.view(-1, self.opt.dim_visual_small+self.opt.dim_spatial)).view(-1, self.opt.image_small_size**2, self.opt.dim_visual_small+self.opt.dim_spatial)
        # (N, 32*32, 832+8) * (N, 832+8, 32*32) -> (N, 32*32, 32*32)
        video_score = F.softmax(torch.bmm(video_key, video_query.permute(0, 2, 1).contiguous()) / np.power(self.opt.dim_visual_small+self.opt.dim_spatial, 0.5), dim=-1)
        # (N, 32*32, 32*32) * (N, 32*32, 832+8) -> (N, 32*32, 832+8)
        weighted_video = torch.bmm(video_score, video_value).view(-1, self.opt.image_small_size, self.opt.image_small_size, self.opt.dim_visual_small+self.opt.dim_spatial)
        # (N, 832+8, 32, 32)
        weighted_video = weighted_video.permute(0, 3, 1, 2).contiguous()

        return weighted_txt, weighted_video


class ACGA(nn.Module):
    def __init__(self, opt):
        super(ACGA, self).__init__()
        self.opt = opt

        self.video_model = VideoEncoder(self.opt)
        self.txt_model = nn.Sequential(
            nn.Conv1d(in_channels=self.opt.dim_semantic, out_channels=self.opt.dim_semantic, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.attention_layer = AttentionLayer(self.opt)

        self.deconv_small = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.opt.dim_visual_small, out_channels=self.opt.dim_visual_medium, kernel_size=8, stride=4, padding=2),
            nn.Conv2d(in_channels=self.opt.dim_visual_medium, out_channels=self.opt.dim_visual_medium, kernel_size=3, padding=1)
        )

        self.conv_mask_small = nn.Sequential(
            nn.Conv2d(in_channels=self.opt.dim_visual_small+self.opt.dim_spatial+self.opt.dim_semantic, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        )

        self.deconv_medium = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.opt.dim_visual_medium, out_channels=self.opt.dim_visual_large, kernel_size=8, stride=4, padding=2),
            nn.Conv2d(in_channels=self.opt.dim_visual_large, out_channels=self.opt.dim_visual_large, kernel_size=3, padding=1)
        )

        self.conv_mask_medium = nn.Sequential(
            nn.Conv2d(in_channels=self.opt.dim_visual_medium+self.opt.dim_spatial+self.opt.dim_semantic, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        )

        self.conv_mask_large = nn.Sequential(
            nn.Conv2d(in_channels=self.opt.dim_visual_large+self.opt.dim_spatial+self.opt.dim_semantic, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        )
        self.conv_fusion = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)

        self.param_init()

    def forward(self, video, txt, spatial_feature_small, spatial_feature_medium, spatial_feature_large):
        # video (N, 528, 4, 32, 32) -> (N, 832, 4, 32, 32)
        video = self.video_model(video.squeeze(1))
        # video_appearance (N, 832, 32, 32), and L2 normalization on video feature
        video = F.normalize(video.mean(-3), p=2, dim=1)
        # txt (N, 300, 20) -> txt_embedding (N, 300, 20)
        txt = self.txt_model(txt)

        txt_attn, video_attn_org = self.attention_layer(video, txt, spatial_feature_small)
        video_attn = video_attn_org[:, :self.opt.dim_visual_small, :, :]

        # branch small
        video_small = torch.cat((video_attn, txt_attn, spatial_feature_small), dim=1)
        response_small = self.conv_mask_small(video_small)

        # deconv small2medium
        # (N, 832, 32, 32) -> (N, 256, 128, 128)
        deconv_small = self.deconv_small(video_attn)
        video_medium = torch.cat((deconv_small, F.interpolate(txt_attn, size=deconv_small.size()[2:], mode='bilinear'), spatial_feature_medium), dim=1)
        response_medium = self.conv_mask_medium(video_medium)

        # deconv medium2large
        # (N, 256, 128, 128) -> (N, 128, 512, 512)
        deconv_medium = self.deconv_medium(deconv_small)
        video_large = torch.cat((deconv_medium, F.interpolate(txt_attn, size=deconv_medium.size()[2:], mode='bilinear'), spatial_feature_large), dim=1)
        response_large = self.conv_mask_large(video_large)

        response_large = self.conv_fusion(torch.cat((F.interpolate(response_small, size=response_large.size()[2:], mode='bilinear'),
                                                     F.interpolate(response_medium, size=response_large.size()[2:], mode='bilinear'),
                                                     response_large), dim=1))
        return response_small.squeeze(1), response_medium.squeeze(1), response_large.squeeze(1)

    def param_init(self):
        self.txt_model.apply(weight_init)
        self.attention_layer.apply(weight_init)
        self.deconv_small.apply(weight_init)
        self.conv_mask_small.apply(weight_init)
        self.deconv_medium.apply(weight_init)
        self.conv_mask_medium.apply(weight_init)
        self.conv_mask_large.apply(weight_init)
        self.conv_fusion.apply(weight_init)
