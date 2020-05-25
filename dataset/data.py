import argparse
import h5py
import pandas as pd

import torch
from torch.utils.data import Dataset


class VideoTextDataset(Dataset):
    def __init__(self, opt):
        super(VideoTextDataset, self).__init__()
        print('loading dataset')

        if not opt.testing:
            fr_name = pd.read_csv(f'{opt.savedir_root}/dataset/{opt.dataset}/preprocessed/train.txt', header=None)
            # (n, 2)
            self.size = torch.from_numpy(fr_name.values[:, 1:3].astype(int))
        else:
            fr_name = pd.read_csv(f'{opt.savedir_root}/dataset/{opt.dataset}/preprocessed/test.txt', header=None)
            # (n, 2)
            self.size = torch.from_numpy(fr_name.values[:, 1:3].astype(int))

        with h5py.File(f'{opt.savedir_root}/dataset/{opt.dataset}/preprocessed/video_{opt.modality}_{opt.version}.h5', 'r', libver='latest', swmr=True) as fr_vid:
            assert fr_vid.swmr_mode
            if not opt.testing:
                self.video = torch.from_numpy(fr_vid['train'][()]).float()
            else:
                self.video = torch.from_numpy(fr_vid['test'][()]).float()

        with h5py.File(f'{opt.savedir_root}/dataset/{opt.dataset}/preprocessed/txt.h5', 'r', libver='latest', swmr=True) as fr_txt:
            assert fr_txt.swmr_mode
            if not opt.testing:
                self.txt = torch.from_numpy(fr_txt['train'][()]).float()
            else:
                self.txt = torch.from_numpy(fr_txt['test'][()]).float()
        # (N, 20, 300) -> (N, 300, 20)
        self.txt = self.txt.permute(0, 2, 1)

        with h5py.File(f'{opt.savedir_root}/dataset/{opt.dataset}/preprocessed/mask.h5', 'r', libver='latest', swmr=True) as fr_mask:
            assert fr_mask.swmr_mode
            if not opt.testing:
                self.mask_large = torch.from_numpy(fr_mask['train/large'][()]).float()
                self.mask_medium = torch.from_numpy(fr_mask['train/medium'][()]).float()
                self.mask_small = torch.from_numpy(fr_mask['train/small'][()]).float()
            else:
                self.mask_large = torch.from_numpy(fr_mask['test/large'][()]).float()
                self.mask_medium = torch.from_numpy(fr_mask['test/medium'][()]).float()
                self.mask_small = torch.from_numpy(fr_mask['test/small'][()]).float()

        with h5py.File(f'{opt.savedir_root}/dataset/{opt.dataset}/preprocessed/bbox.h5', 'r', libver='latest', swmr=True) as fr_bbox:
            assert fr_bbox.swmr_mode
            if not opt.testing:
                self.bbox_large = torch.from_numpy(fr_bbox['train/large'][()]).float()
                self.bbox_medium = torch.from_numpy(fr_bbox['train/medium'][()]).float()
                self.bbox_small = torch.from_numpy(fr_bbox['train/small'][()]).float()
            else:
                self.bbox_large = torch.from_numpy(fr_bbox['test/large'][()]).float()
                self.bbox_medium = torch.from_numpy(fr_bbox['test/medium'][()]).float()
                self.bbox_small = torch.from_numpy(fr_bbox['test/small'][()]).float()

        self.ndata = self.txt.shape[0]

    def __getitem__(self, item):
        return self.size[item], self.video[item], self.txt[item], \
               self.mask_small[item], self.mask_medium[item], self.mask_large[item], \
               self.bbox_small[item], self.bbox_medium[item], self.bbox_large[item]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ndata)[:batch_size]
        return self.size[idx], self.video[idx], self.txt[idx], \
               self.mask_small[idx], self.mask_medium[idx], self.mask_large[idx], \
               self.bbox_small[idx], self.bbox_medium[idx], self.bbox_large[idx]

    def __len__(self):
        return self.ndata
