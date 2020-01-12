"""
Ultrasound Nerve Segmentation: Datasets, Transforms, training scripts,
validation and prediction methods.

Third version of USNS detector. Realized as class, containing
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


## Transform

img_mean = (0.38983212684516944,)
img_std = (0.21706658034222048,)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=img_mean, std=img_std)
    ])

transform_mask = T.Compose([T.ToTensor()])


## Dataset

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


## Trainer-container :)

class USNSDetector:
    def __init__(self, model, device=None, eps_dice=1e-6):
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

        self.eps_dice = eps_dice
        
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_model_dice = 0.0

    def fit(self, dataloader, dataloader_val, epochs=1, print_every=1000, lr=0.001):
        """
        Train model.
        Args:
            dataloader: pytorch compatible DataLoader, that provides
                minibatches of input preprocessed images and corresponding
                labled target images
            dataloader_val: pytorch compatible DataLoader for validation
            epochs: number of epochs to train
            print_every: num of samples to be processed to check val score
            params: dict with training hyperparameters
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.n_samples = []
        self.loss_history = []
        self.val_loss_history = []
        self.dice_history = []

        running_loss = 0
        num_images = 0
        num_images_previous = 0
        for ee in range(epochs):
            for x, y in dataloader:
                self.model.train()
                x = x.to(self.device)
                y = y.to(self.device)
                y = y.squeeze().long()

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

                    if dscore > self.best_model_dice:
                        self.best_model_dice = dscore
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())

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
        scores = F.pad(scores, (2, 2, 2, 2))
        loss = F.cross_entropy(scores, target)
        
        return loss

    def dice(self, pred, target):
        """
        Dice coef as scored in competition. Accepts predictions (0 or 1 for
        any pixel), not scores, so should not be used in loss function
        """

        # both_empty == 1 only if both pred and target are empty, 0 in other cases
        both_empty = (1 - pred.flatten(start_dim=1).max(dim=1)[0]) * \
                     (1 - target.flatten(start_dim=1).max(dim=1)[0])

        D = (2 * torch.sum(pred * target, (-2, -1)) + both_empty).float() / \
            (torch.sum(pred + target, (-2, -1)) + both_empty).float()

        return D
        
    def dice_loss(self, scores, target):
        """
        Dice loss finction. Should be minimized! Accepts raw scores without
        softmax or sigmoid.
        args:
            scores: model output scores
            target:
        """
        scores = scores.flatten(start_dim=2)
        scores = F.softmax(scores, dim=1)
        target = scores.flatten(start_dim=1)

        Dpos = (torch.sum(scores[:, 1, :] * target, dim=-1) + self.eps_dice) / \
            (torch.sum(scores[:, 1, :] + target, dim=-1) + self.eps_dice)
        Dneg = (torch.sum(scores[:, 0, :] * (1 - target), dim=-1) + self.eps_dice) / \
            (torch.sum(scores[:, 0, :] + 1 - target, dim=-1) + self.eps_dice)
        
        loss = 1 - Dpos - Dneg

        return loss

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
                x = x.to(self.device)
                y = y.to(self.device)
                y = y.squeeze().long()

                batch_size = y.shape[0]
                scores = self.model(x)
                loss += batch_size * self.loss(scores, y).item()

                pred = torch.argmax(scores, dim=1)
                pred = F.pad(pred, (2, 2, 2, 2))
                dice += torch.sum(self.dice(pred, y))

                n_processed += batch_size
            loss = loss / n_processed
            dice = dice / n_processed

            if printing:
                print(f'Validation loss = {loss}')
                print(f'Val Dice score = {dice}')
            
            if show:
                show_imgs(x.cpu(),
                          y.cpu().unsqueeze(1),
                          pred.float().cpu().unsqueeze(1))
        
        return loss, dice

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
                        out_mask = pred[nn, :, :]
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
            scores = self.model(x)
            preds = torch.argmax(scores, dim=1)
            preds = F.pad(preds, (2, 2, 2, 2))
            preds = preds.float()

        return preds


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