"""
Ultrasound Nerve Segmentation all in one: CNN with training scripts,
validation and prediction methods.

Second (or even third) version of USNS detector. Realized as class, containing
all needed procedures. It is aimed to simplyfy hyperparameter exchange between
diffferent methods or functions.
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt


img_mean = (0.38983212684516944,)
img_std = (0.21706658034222048,)


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



## Net definition

class ConvBlockDn(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0):
        """
        Convolution block of down pass in Unet
        """
        super(ConvBlockDn, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), (2, 2))
        x = F.relu(x)

        return x


class ConvBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0):
        """
        Convolution block of down pass in Unet
        """
        super(ConvBlockUp, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.upconv = nn.ConvTranspose2d(out_channel, out_channel // 2,
                                         (2, 2), stride=2, bias=True)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upconv(x)

        return x


class Unet(nn.Module):
    def __init__(self, n_filters=64):
        """
        Unet CNN for Ultrasound Nerve Segmentation challenge
        """
        super(Unet, self).__init__()

        # down
        self.dn_blk1 = ConvBlockDn(1, n_filters, pad=0)
        self.dn_blk2 = ConvBlockDn(n_filters, 2*n_filters, pad=1)
        self.dn_blk3 = ConvBlockDn(2*n_filters, 4*n_filters, pad=1)
        self.dn_blk4 = ConvBlockDn(4*n_filters, 8*n_filters, pad=1)

        # bottom
        self.bottom_blk = ...

        # up
        self.up_blk1 = ConvBlockUp(8*n_filters, 4*n_filters)
        self.up_blk2 = ConvBlockUp(4*n_filters, 2*n_filters)
        self.up_blk3 = ConvBlockUp(2*n_filters, n_filters)

    def forward(self, x):
        # go down

        # bottom block

        # go up

        # out block


class USNSDetector:
    def __init__(self, model:nn.Module, device=None):
        """
        Ultrasound Nerve Segmentation detector class. Contains training and
        prediction procedures.
        Args:
            model: (nn.Module) model to use or train
            device: (torch.devive) device to use for major computations
        """
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = model.to(self.device)

        self.threshold = 0.5

    def fit(self, dataloader, dataloader_val, epochs=1, print_every=1000, **params):
        """
        Train model.
        Args:
            dataloader: pytorch compatible DataLoader, that provides
                minibatches of input preprocessed images and corresponding
                labled target images
            dataloader_val: pytorch compatible DataLoader for validation
            epochs: number of epochs to train
            optimizer: pytorch compatible optimiser to train with
            params: dict with training hyperparameters
            print_every: num of samples to be processed to check val score
        """
        optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])
        
        running_loss = 0
        num_images = 0
        self.loss_history = []
        self.val_loss_history = []
        self.dice_history = []
        for ee in range(epochs):
            for x, y in dataloader:
                self.model.train()
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.shape[0]

                scores = self.model(x)
                loss = self.loss(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += batch_size * loss.item()
                num_images += batch_size

                if num_images % print_every < batch_size:
                    self.loss_history.append(running_loss / num_images)
                    print(f'{num_images} images processed, loss = {self.loss_history[-1]}')
                    
                    vloss, dscore = self.validate(dataloader_val, printing=True)
                    self.val_loss_history.append(vloss)
                    self.dice_history.append(dscore)
                    print()

            print(f'After {ee+1} epochs: loss = {running_loss / max(nn, 1e-6)}')
            self.validate(dataloader_val, printing=True, show=True)
            print()

    def loss(self, scores, target):
        """
        Loss function used for base model trainig
        args:
            scores: model output scores
            target: ground truth lables
        returns:
            scalar loss for input minibatch
        """
        loss = F.binary_cross_entropy_with_logits(scores,
                                                  target[:, :, 2:-2, 2:-2],
                                                  reduction='mean')
        return loss
    
    def validate(self, dataloader, printing=False, show=True):
        """
        Scores model with loss and Dice score for data from dataloader
        Args:
            dataloader: Pytorch DataLoader (with groubd truth)
            printing: (bool) if to print loss and score (default: False)
            show: (bool) if to show input, ground truth and scores for last
                batch from dataloader
        Returns:
            (loss, dice) tuple or loss and dice score for data
        """
        self.model.eval()

        nn = 0
        loss = 0
        dice = 0
        with torch.no_grad():
            for x, y in dataloader:
                batch_size = y.shape[0]
                scores = self.model(x.to(self.device))
                loss += batch_size * self.loss(scores, y.to(self.device))
                dice += np.sum(self.score(scores, y))
                nn += batch_size
            loss = loss / nn
            dice = dice / nn

            if printing:
                print(f'Validation loss = {loss}')
                print(f'Dice score = {dice}')
            
            if show:
                show_imgs(x.cpu(),
                          y.cpu(),
                          torch.sigmoid(scores.cpu()))
        
        return loss, dice

    def score(self, scores, targets):
        """
        Calculate Dice score for validation
        Args:
            scores: torch minibatch of model predictions (model outputs without
                logits)
            targets: torch minibatch of grount truth mask
    
        Returns:
            numpy array of dice score for each minibatch
        """
        with torch.no_grad():
            batch_size = scores.shape[0]
            # raw scores to probability
            scores = torch.sigmoid(scores)
            scores = F.pad(scores, (2, 2, 2, 2))
            
            scores = scores.cpu().numpy()
            targets = targets.cpu().numpy()

            scores = scores > 0.5
            
            dice = np.zeros(batch_size)
            for nn in range(batch_size):
                dice[nn] = dice_coef(scores[nn, 0, :, :], targets[nn, 0, :, :])
        
        return dice
    
    def generate_submission(self, dataloader, filepath):
        """
        Generate submission csv file for samples from dataloader
        """
        with torch.no_grad():
            with open(filepath, mode='w') as submission:
                submission.write('img,pixels\n')
                for x, y in dataloader:
                    x = x.to(self.device)
                    pred = self.__call__(x)
                    for nn in range(pred.shape[0]):
                        out_mask = pred[nn, :, :, :]
                        rle = rle_encoding(out_mask.cpu().numpy())
                        img_item = int(y[nn].split('_')[0])
                        
                        submission.write(str(img_item) + ',')
                        for rr in rle:
                            submission.write(str(rr) + ' ')
                        submission.write('\n')
    
    def __call__(self, x):
        
        """
        Evaluate model with input x.
        Returns predicted mask image
        """
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(x))
            scores = F.pad(scores, (2, 2, 2, 2))
            out_mask = scores > self.threshold
            out_mask = out_mask.float()

        return out_mask


def dice_coef(pred, targ):
    """
    Dice score for prediction image pred with ground truth image targ
    """
    pred = pred.astype(bool)
    targ = targ.astype(bool)
    
    if targ.any():
        D = 2 * np.sum((targ * pred).astype(float))
        D = D / (np.sum(targ.astype(float)) + np.sum(pred.astype(float)))
        return D
    else:
        return float(~pred.any())


def show_imgs(*imgs):
    batch_size = imgs[0].shape[0]
    f, axarr = plt.subplots(batch_size, len(imgs), figsize=(16, 4 * batch_size), squeeze=False)
    with torch.no_grad():
        for nn in range(batch_size):
            for ii, img in enumerate(imgs):
                axarr[nn, ii].axis('off')
                axarr[nn, ii].imshow(img[nn, 0])
    f.show()


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