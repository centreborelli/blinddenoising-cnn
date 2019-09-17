import torch
import torch.nn as nn
from torch.utils.data import Dataset

def smartfilter(sigma=None, maxsize=100):
    import numpy as np
    past = []
    def filter(pair):
        nonlocal past
        apast = np.array(past)
        dist = torch.dist(pair[0], pair[1]).item()
        past.append(dist)
        if len(past) > maxsize:
            past = past[-maxsize:]
        if sigma and len(past) > maxsize // 2:
            med = np.median(apast)
            th = med + sigma * np.median(np.abs(med - apast))
            if dist > th:
                return False
        return True
    return filter

def crop(pair, cropsize):
    size = pair[0].size()
    w, h = size[-2], size[-1]
    x = random.randint(0, w - cropsize)
    y = random.randint(0, h - cropsize)
    return tuple(img[...,x:x+cropsize,y:y+cropsize] for img in pair)

import random
def add_noise(pair, sigma):
    if sigma < 0:
        sigma = random.random() * -sigma
    return tuple((img + torch.randn_like(img) * sigma) for img in pair)

def read(filename):
    import imageio
    import numpy as np
    try:
        img = imageio.imread(filename).astype('float32') / 255
    except:
        print(filename)
    img = np.atleast_3d(img)
    v = torch.from_numpy(img)
    v = v.permute((2,0,1))
    return v


class FolderFusionDataset(Dataset):

    def __init__(self, root, skip=1500):
        super().__init__()
        self.root = root
        import os
        from natsort import natsorted
        self.imgs = natsorted(os.listdir(self.root))[skip:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return read(f'{self.root}/{self.imgs[idx]}')


class LayerConcatDataset(Dataset):

    def __init__(self, datasets, roll=0):
        super().__init__()
        self.datasets = tuple(datasets)
        self.roll = roll
        self.n = min(len(ds) for ds in self.datasets)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        v = torch.cat([ds[idx] for ds in self.datasets], dim=0)
        v = v.roll(shifts=self.roll, dims=0)
        return v


class TwoFramesDataset(Dataset):

    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert(len(self.dataset1) == len(self.dataset2))

    def __len__(self):
        return len(self.dataset1) - 1

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx + 1]


class ConcatDataset(Dataset):

    def __init__(self, datasets):
        super().__init__()
        self.datasets = tuple(datasets)
        self.n = sum(len(ds) for ds in self.datasets)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx < len(ds):
                return ds[idx]
            idx -= len(ds)
        print(idx)
        assert(False)


class AugmentedData(Dataset):

    def __init__(self, dataset, function):
        super().__init__()
        self.dataset = dataset
        self.function = function

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.function(self.dataset[idx])

