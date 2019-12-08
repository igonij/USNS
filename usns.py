import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt

img_mean = (0.38983212684516944,)
img_std = (0.21706658034222048,)

device = torch.device('cuda')
dtype = torch.cuda.FloatTensor

# Run-length encoding function (https://www.kaggle.com/rakhlin/fast-run-length-encoding-python)
def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    source: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    """
    dots = np.where(x.T.flatten() == 1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=img_mean, std=img_std)
    ])

transform_mask = T.Compose([T.ToTensor()])

class USNSDataset(Dataset):
    def __init__(self,
                 datadir,
                 train=True,
                 transform=T.ToTensor(),
                 mask_transform=T.ToTensor()):
        """
        Dataset class for loading images and masks.
        Data as in https://www.kaggle.com/c/ultrasound-nerve-segmentation/data
            datadir (string): Path to images
            train (bool): if True, read also a ground truth mask. Otherwise,
            mask filename transform (callable): transform to be applied on a
            sample
        """
        self.datadir = datadir
        self.train = train
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.imglist = [ii for ii in os.listdir(datadir) if 'mask' not in ii]
        self.imglist.sort(key=lambda fname: tuple(map(int,
                                            fname.split('.')[0].split('_'))))
        
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, index):
        img_fname = self.imglist[index]
        mask_fname = img_fname.replace('.tif', '_mask.tif')
        
        img = Image.open(os.path.join(self.datadir, img_fname))
        img = np.expand_dims(img, axis=2)
        if self.train:
            mask = Image.open(os.path.join(self.datadir, mask_fname))
            mask = np.expand_dims(mask, axis=2)
            mask = np.array(mask)
            mask = mask / 255
        else:
            mask = mask_fname
        
        if self.transform:
            img = self.transform(img)
            if self.train:
                mask = self.mask_transform(mask)
        
        return (img, mask)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0):
        """
        Convolutional block of U-net architecture
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        return x

class Unet(nn.Module):
    def __init__(self, n_filters=64):
        super().__init__()
        
        self.n_filters = n_filters
        
        # down
        self.down_block1 = ConvBlock(1, n_filters, pad=0)
        self.down_block2 = ConvBlock(n_filters, 2*n_filters, pad=1)
        self.down_block3 = ConvBlock(2*n_filters, 4*n_filters, pad=1)
        self.down_block4 = ConvBlock(4*n_filters, 8*n_filters, pad=1)
        
        # bottom
        self.bottom_block = ConvBlock(8*n_filters, 16*n_filters, pad=1)
        
        # up (there are learning params in up conv)
        self.upconv1 = nn.ConvTranspose2d(16*n_filters, 8*n_filters,
                                          (2, 2), stride=2, bias=True)
        self.up_block1 = ConvBlock(16*n_filters, 8*n_filters, pad=1)
        self.upconv2 = nn.ConvTranspose2d(8*n_filters, 4*n_filters,
                                          (2, 2), stride=2, bias=True)
        self.up_block2 = ConvBlock(8*n_filters, 4*n_filters, pad=1)
        self.upconv3 = nn.ConvTranspose2d(4*n_filters, 2*n_filters,
                                          (2, 2), stride=2, bias=True)
        self.up_block3 = ConvBlock(4*n_filters, 2*n_filters, pad=1)
        self.upconv4 = nn.ConvTranspose2d(2*n_filters, n_filters,
                                          (2, 2), stride=2, bias=True)
        self.up_block4 = ConvBlock(2*n_filters, n_filters, pad=1)
        
        self.outconv = nn.Conv2d(n_filters, 1, (1, 1), bias=True)
    
    def forward(self, x):
        # go down
        out_dn1 = self.down_block1(x)
        x = F.max_pool2d(out_dn1, (2, 2))
        out_dn2 = self.down_block2(x)
        x = F.max_pool2d(out_dn2, (2, 2))
        out_dn3 = self.down_block3(x)
        x = F.max_pool2d(out_dn3, (2, 2))
        out_dn4 = self.down_block4(x)
        x = F.max_pool2d(out_dn4, (2, 2))
        
        # bottom block
        x = self.bottom_block(x)
        
        # go up (like here https://arxiv.org/pdf/1505.04597.pdf,
        # but using transpose conv to preserve dimensions in output)
        x = self.upconv1(x)
        x = torch.cat([out_dn4, x], dim=1)
        x = self.up_block1(x)
        
        x = self.upconv2(x)
        x = torch.cat([out_dn3, x], dim=1)
        x = self.up_block2(x)
        
        x = self.upconv3(x)
        x = torch.cat([out_dn2, x], dim=1)
        x = self.up_block3(x)
        
        x = self.upconv4(x)
        x = torch.cat([out_dn1, x], dim=1)
        x = self.up_block4(x)
        
        x = self.outconv(x)
        
        return x


def train(model, optimizer, dataloader, dataloader_val, epochs=1, print_every=1000):
    """
    Train a model with optimizer using data provided by dataloader.
        model: PyTorch nn.Module
        optimiser: Pytorch torch.optim with tuned learning hyperparameters
        dataloader: torch.utils.data.DataLoader
        epochs: number of epochs to train
        print_every: Print loss and check accuracy every print_every iterations
    """
    
    for ee in range(epochs):
        for t, (x, y) in enumerate(dataloader):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            loss = F.binary_cross_entropy_with_logits(scores,
                                                      y[:, :, 2:-2, 2:-2],
                                                      reduction='mean')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_acc += loss
            if t % print_every == print_every - 1:
                print(f'Iteration {t}, loss = {loss.item()}')
                val_loss(model, dataloader_val)
                print()
        print(f'After {ee+1} epochs: loss = {loss.item()}')
        val_loss(model, dataloader_val, show=True)
        print()


def val_loss(model, dataloader, show=False):
    """
    Check accuracy of the model on validation dataset
    """
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            loss = F.binary_cross_entropy_with_logits(scores,
                    y[:, :, 2:-2, 2:-2], reduction='mean')
        
        print(f'Validation loss = {loss.item()}')
        
        if show:
            show_imgs([x[0, 0].cpu(),
                       y[0, 0].cpu(),
                       torch.sigmoid(scores[0, 0].cpu())])


def eval_model(model, x, threshold=0.5):
    """
    Evaluate model with input x.
    Returns predicted mask image
    """
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x))
        scores = F.pad(scores, (2, 2, 2, 2))
        out_mask = scores > threshold
        out_mask = out_mask.int()

    return out_mask


def dice_coef(pred, y):
    pred = pred.astype(bool)
    y = y.astype(bool)
    
    D = y * pred + (~y) * (~pred)
    
    return D.mean()


def generate_submission(model, dataloader, filepath):
    with torch.no_grad():
        model.to(device=device)
        with open(filepath, mode='w') as submission:
            submission.write('img,pixels\n')
            
            for t, (x, y) in enumerate(dataloader):
                x = x.to(device=device)
                pred = eval_model(model, x, threshold=0.5)
                for nn in range(pred.shape[0]):
                    out_mask = pred[nn, :, :, :]
                    rle = rle_encoding(out_mask.cpu().numpy())
                    img_item = int(y[nn].split('_')[0])
                    
                    submission.write(str(img_item) + ',')
                    for rr in rle:
                        submission.write(str(rr) + ' ')
                    submission.write('\n')


def show_imgs(imglist):
    f, axarr = plt.subplots(1, len(imglist))
    with torch.no_grad():
        for ii, data in enumerate(imglist):
            axarr[ii].axis('off')
            axarr[ii].imshow(data)
    plt.show()
