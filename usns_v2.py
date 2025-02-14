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

import copy

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

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0, bn=False):
        """
        Convolutional block of U-net architecture without activation (it is
        optimal to make ReLU after max pool)
        """
        super(ConvBlock, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        
        if self.bn:
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.bn: x = self.bn1(x)
        x = self.conv2(x)
        if self.bn: x = self.bn2(x)
        
        return x


class DnPool(nn.Module):
    """
    Down pass Max Pooling
    """
    def __init__(self):
        super(DnPool, self).__init__()
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(x, (2, 2)))

        return x

class UpPool(nn.Module):
    """
    Up convolution on the way up
    Acceprs input x from previouse layer and concatenates output with
    features f from down pass
    """
    def __init__(self, in_channel):
        super(UpPool, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channel, in_channel // 2,
                                         (2, 2), stride=2, bias=True)
    
    def forward(self, x, f):
        x = self.upconv(F.relu(x))
        # do we need relu for x here?
        out = F.relu(torch.cat([f, x], dim=1))

        return out 


class Unet(nn.Module):
    def __init__(self, n_filters=64, bn=False):
        """
        Unet CNN for Ultrasound Nerve Segmentation challenge
            n_filters: (int) initial number of filters
            bn: (bool)  yse of batch normalization
        """
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=0, bn=bn)
        self.dnpool1 = DnPool()
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dnpool2 = DnPool()
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dnpool3 = DnPool()
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        self.dnpool4 = DnPool()

        # bottom
        self.bottom_blk = ConvBlock(8*n_filters, 16*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(16*n_filters)
        self.up_blk1 = ConvBlock(16*n_filters, 8*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(8*n_filters)
        self.up_blk2 = ConvBlock(8*n_filters, 4*n_filters, pad=1, bn=bn)
        self.upconv3 = UpPool(4*n_filters)
        self.up_blk3 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv4 = UpPool(2*n_filters)
        self.up_blk4 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)

        self.outconv = nn.Conv2d(n_filters, 1, (1, 1), bias=True)

    def forward(self, x):
        # go down
        out1 = self.dn_blk1(x)
        out2 = self.dn_blk2(self.dnpool1(out1))
        out3 = self.dn_blk3(self.dnpool2(out2))
        out4 = self.dn_blk4(self.dnpool1(out3))
        
        # bottom block
        self.bottom_out = self.bottom_blk(self.dnpool4(out4))
        
        # go up
        x = self.up_blk1(self.upconv1(self.bottom_out, out4))
        x = self.up_blk2(self.upconv2(x, out3))
        x = self.up_blk3(self.upconv3(x, out2))
        x = self.up_blk4(self.upconv4(x, out1))

        # out block
        x = self.outconv(x)

        return x


class BinaryNet(nn.Module):
    """
    Binary classifier deciding if nerve present on image
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()
        self.bn = bn

        self.conv1 = nn.Conv2d(16*n_filters, 8*n_filters, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(8*n_filters, 4*n_filters, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(4*n_filters, 2*n_filters, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(2*n_filters, n_filters, kernel_size=(3, 3))
        self.conv1x1 = nn.Conv2d(n_filters, 1, kernel_size=(1, 1))
        self.fc = nn.Linear(504, 1)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(8*n_filters)
            self.bn2 = nn.BatchNorm2d(4*n_filters)
            self.bn3 = nn.BatchNorm2d(2*n_filters)
            self.bn4 = nn.BatchNorm2d(n_filters)
    
    def forward(self, x):
        x = self.conv1(F.relu(x))
        if self.bn: x = self.bn1(x)

        x = self.conv2(F.relu(x))
        if self.bn: x = self.bn2(x)

        x = self.conv3(F.relu(x))
        if self.bn: x = self.bn3(x)
        
        x = self.conv4(F.relu(x))
        if self.bn: x = self.bn4(x)

        x = self.conv1x1(F.relu(x))
        x = self.fc(F.relu(x.flatten(start_dim=1)))

        return x


class BinaryNetSimple(nn.Module):
    """
    Binary classifier deciding if nerve present on image
    """
    def __init__(self, n_filters):
        super().__init__()
        
        self.conv1x1 = nn.Conv2d(16*n_filters, 1, kernel_size=(1, 1))
        self.fc = nn.Linear(936, 1)

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = self.fc(x.flatten(start_dim=1))

        return x


class USNSDetector:
    def __init__(self, model, model_bin, device=None):
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
        self.model_bin = model_bin.to(self.device)

        self.threshold = torch.tensor(0.5).to(self.device)
        self.threshold.requires_grad_()

        self.n_samples = []
        self.loss_history = []
        self.val_loss_history = []
        self.dice_history = []

        self.n_samples_bin = []
        self.loss_history_bin = []
        self.val_loss_history_bin = []

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_model_dice = 0.0

        self.best_model_bin_wts = copy.deepcopy(self.model_bin.state_dict())
        self.best_model_acc = 0.0
        
    def fit(self, dataloader, dataloader_val, epochs=1, print_every=1000, lr=0.001):
        """
        Train model.
        Args:
            dataloader: pytorch compatible DataLoader, that provides
                minibatches of input preprocessed images and corresponding
                labled target images
            dataloader_val: pytorch compatible DataLoader for validation
            epochs: number of epochs to train
            optimizer: pytorch compatible optimiser to train with
            print_every: num of samples to be processed to check val score
            params: dict with training hyperparameters
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        optimizer.add_param_group({'params': self.threshold})
        
        running_loss = 0
        num_images = 0
        num_images_previous = 0
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
                    self.n_samples.append(num_images)
                    self.loss_history.append(running_loss / (num_images - num_images_previous))
                    running_loss = 0
                    num_images_previous = num_images
                    print(f'{num_images} images processed, loss = {self.loss_history[-1]}')
                    
                    vloss, dscore = self.validate(dataloader_val, printing=True)
                    self.val_loss_history.append(vloss)
                    self.dice_history.append(dscore)
                    print()

            print(f'After {ee+1} epochs: loss = {running_loss / max(num_images, 1e-6)}')
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

    def dice(self, scores, target):
        scores = F.pad(scores, (2, 2, 2, 2)).sigmoid()
        scores = F.relu(scores - self.threshold)
        
        
    
    def validate(self, dataloader, printing=False, show=False):
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

        n_processed = 0
        loss = 0
        dice = 0
        with torch.no_grad():
            for x, y in dataloader:
                batch_size = y.shape[0]
                scores = self.model(x.to(self.device))
                loss += batch_size * self.loss(scores, y.to(self.device))
                dice += np.sum(self.score(scores, y))
                n_processed += batch_size
            loss = loss / n_processed
            dice = dice / n_processed

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
    
    def fit_binary(self, dataloader_train, dataloader_val,
                        epochs=1, print_every=1000, lr=0.001):
        
        optimizer = optim.Adam(self.model_bin.parameters(), lr=lr)

        running_loss = 0
        num_images = 0
        num_images_previous = 0
        for ee in range(epochs):
            for x, y in dataloader_train:
                self.model.eval()
                self.model_bin.train()
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.shape[0]

                scores = self.model(x)
                x = self.model.bottom_out.detach()
                scores = self.model_bin(x)
                loss = self.loss_bin(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += batch_size * loss.item()
                num_images += batch_size

                if num_images % print_every < batch_size:
                    self.n_samples_bin.append(num_images)
                    self.loss_history_bin.append(running_loss / (num_images - num_images_previous))
                    running_loss = 0
                    num_images_previous = num_images
                    print(f'{num_images} images processed, loss = {self.loss_history_bin[-1]}')
                    
                    vloss = self.validate_bin(dataloader_val, printing=True)
                    self.val_loss_history_bin.append(vloss)
                    print()

            print(f'After {ee+1} epochs: loss = {running_loss / max(num_images - num_images_previous, 1e-6)}')
            self.validate_bin(dataloader_val, printing=True)
            print()
    
    def loss_bin(self, scores, target):
        target = target.flatten(start_dim=1).bool().any(dim=1)
        target = target.float()
        loss = F.binary_cross_entropy_with_logits(scores.squeeze(), target)
        return loss

    def validate_bin(self, dataloader, printing=False):
        self.model.eval()
        self.model_bin.eval()

        n_processed = 0
        loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                batch_size = y.shape[0]
                scores = self.model(x.to(self.device))
                scores = self.model.bottom_out.detach()
                scores = self.model_bin(scores)
                loss += batch_size * self.loss_bin(scores, y.to(self.device)).item()
                n_processed += batch_size
            loss = loss / n_processed
            
            if printing:
                print(f'Validation loss = {loss}')
            
        return loss
    
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
        self.model.eval()
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
    _, axarr = plt.subplots(batch_size, len(imgs), figsize=(16, 4 * batch_size), squeeze=False)
    with torch.no_grad():
        for nn in range(batch_size):
            for ii, img in enumerate(imgs):
                axarr[nn, ii].axis('off')
                axarr[nn, ii].imshow(img[nn, 0])
    plt.show()


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