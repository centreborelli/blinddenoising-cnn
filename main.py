#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DnCNN(nn.Module):

    def __init__(self, inchannels, outchannels=1, num_of_layers=17):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=inchannels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        d = self.dncnn(x)
        if self.inchannels > self.outchannels:
            x = x[..., :self.outchannels, :, :]
        return x + d

    def init_weights(self):
        def weights_init_kaiming(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(weights_init_kaiming)


def get_folders(root):
    import glob
    return sorted(glob.glob(f'{root}/*/take_*/'))

def dataaugmenter(sigma, cropsize, inchannels, outchannels):
    filterer = data.smartfilter(sigma=3)
    def a(pair):
        c = None
        while c is None or not filterer(c):
            c = data.crop(pair, cropsize)
        c = list(c)
        c[0] = c[0][..., :inchannels, :, :]
        c[1] = c[1][..., :outchannels, :, :]
        return c
    return a

def dataaugmenter2(inchannels, outchannels):
    def a(pair):
        c = list(pair)
        c[0] = c[0][..., :inchannels, :, :]
        c[1] = c[1][..., :outchannels, :, :]
        return c
    return a

def write_tensor(path, tensor):
    import iio
    tensor = tensor.permute((0, 2, 3, 1)).squeeze()
    iio.write(path, tensor.cpu().detach().numpy())

def train(root,
          invis='fusion20', outvis='fusion20',
          inir='fusion', outir='fusion',
          predict='vis',
          sigma=0, cropsize=200,
          inchannels=1, outchannels=1,
          preload=None, saveto=None, batchsize=20,
          lr=1e-3, nbepochs=60):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = DnCNN(inchannels=inchannels, outchannels=outchannels)
    net.init_weights()
    if preload:
        net.load_state_dict(torch.load(preload))
    net.cuda()

    folders = get_folders(root+'fusion')
    da = dataaugmenter(sigma, cropsize, inchannels, outchannels)
    ds_train = AugmentedData(
        ConcatDataset(
            TwoFramesDataset(
                LayerConcatDataset([
                    FolderFusionDataset(folder.replace('fusion', invis) + '/VIS'),
                    FolderFusionDataset(folder.replace('fusion', inir) + '/IR'),
                ], roll=0 if predict == 'vis' else 1),
                LayerConcatDataset([
                    FolderFusionDataset(folder.replace('fusion', outvis) + '/VIS'),
                    FolderFusionDataset(folder.replace('fusion', outir) + '/IR'),
                ], roll=0 if predict == 'vis' else 1),
            ) for folder in folders[1:]),
        da)

    da = dataaugmenter2(inchannels, outchannels)
    ds_val = AugmentedData(
        ConcatDataset(
            TwoFramesDataset(
                LayerConcatDataset([
                    FolderFusionDataset(folder.replace('fusion', invis) + '/VIS'),
                    FolderFusionDataset(folder.replace('fusion', inir) + '/IR'),
                ], roll=0 if predict == 'vis' else 1),
                LayerConcatDataset([
                    FolderFusionDataset(folder.replace('fusion', 'fusion') + '/VIS'),
                    FolderFusionDataset(folder.replace('fusion', 'fusion') + '/IR'),
                ], roll=0 if predict == 'vis' else 1),
            ) for folder in folders[:1]),
        da)

    loader_train = DataLoader(dataset=ds_train, num_workers=2, batch_size=batchsize, shuffle=True)
    loader_val = DataLoader(dataset=ds_val, num_workers=2, batch_size=1, shuffle=False)

    criterion = nn.MSELoss()
    criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5,
                           amsgrad=False, eps=1e-8, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    def do_batch(noisy, target, train=True):
        if train:
            optimizer.zero_grad()
        denoised = net(noisy.to(device))
        loss = criterion(denoised, target.to(device))
        if train:
            loss.backward()
            optimizer.step()
            # if scheduler:
                # scheduler.step()
        return denoised, loss

    from progressbar import progressbar as pb
    for epoch in range(nbepochs):
        net.train()
        for i, (noisy, target) in pb(enumerate(loader_train)):
            denoised, loss = do_batch(noisy, target)
            if i % 100 == 0:
                write_tensor('noisy.tif', noisy[0:1,...])
                write_tensor('target.tif', target[0:1,...])
                write_tensor('denoised.tif', denoised[0:1,...])
            del loss
            del denoised

        net.eval()
        with torch.no_grad():
            l = 0
            n = 0
            for i, (noisy, target) in enumerate(loader_val):
                if i > 3: break
                denoised, loss = do_batch(noisy, target, train=False)
                l += loss.item()
                write_tensor(f'{saveto}/val_{epoch}_{i}_noisy.tif', noisy[0:1,...])
                write_tensor(f'{saveto}/val_{epoch}_{i}_target.tif', target[0:1,...])
                write_tensor(f'{saveto}/val_{epoch}_{i}_denoised.tif', denoised[0:1,...])
                n += 1
            print('val:', epoch, l/n)
        scheduler.step(l)

        if saveto:
            torch.save([net, optimizer], f'{saveto}/checkpoint_{epoch}.tar')

def test(root, outputdir,
         invis='fusion20', inir='fusion',
         predict='vis',
         model=None):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = torch.load(model)[0]
    net.cuda()

    dirvis = root.replace('fusion', invis)+'/VIS'
    dirir = root.replace('fusion', inir)+'/IR'
    from progressbar import progressbar as pb
    for vis, ir in pb(zip(sorted(os.listdir(dirvis)),
                          sorted(os.listdir(dirir)))):
        f = vis
        vis = f'{dirvis}/{vis}'
        ir = f'{dirir}/{ir}'

        imgvis = data.read(vis)
        imgir = data.read(ir)
        input = torch.cat([imgvis, imgir], dim=0)

        if predict != 'vis':
            input = input.roll(shifts=1, dims=0)

        input = input[:net.inchannels, ...]
        input = input.unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(input)

        write_tensor(f'{outputdir}/{f}.tif', out*255)

def mse_to_psnr(mse):
    import math
    return 10 * math.log10(1 / mse)

def psnr(root, outputdir,
         predict='vis'):
    criterion = nn.MSELoss()

    d = 'VIS'
    if predict == 'ir':
        d = 'IR'
    gtdir = root + '/' + d + '/'

    filt = lambda l: sorted(filter(lambda x: not x.endswith('.psnr'), l))

    from progressbar import progressbar as pb
    for gt, pred in pb(zip(filt(os.listdir(gtdir)),
                           filt(os.listdir(outputdir)))):
        pred = f'{outputdir}/{pred}'
        groundtruth = data.read(f'{gtdir}/{gt}')
        predicted = data.read(pred)

        mse = criterion(groundtruth, predicted).item()
        psnr = mse_to_psnr(mse)
        open(pred + '.psnr', 'w').write(f'{psnr:.3f}')

if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'test': test, 'psnr': psnr})

