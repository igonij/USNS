"""
Ultrasound Nerve Segmentation: Datasets, Transforms, training scripts,
validation and prediction methods.

Third version of USNS detector. Realized as class, containing
all needed procedures. It is aimed to simplyfy hyperparameter exchange between
diffferent methods or functions.
"""

import os
import copy
import random
import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image

from matplotlib import pyplot as plt


## Transforms

class Resize(T.Resize):
    """Resize transform redefinition
    """
    def __init__(self, size, interpolation=Image.BILINEAR, resize_mask=True):
        super().__init__(size, interpolation)
        self.resize_mask = resize_mask

    def __call__(self, imglist):
        assert len(imglist) <= 2
        imglist[0] = TF.resize(imglist[0], self.size, self.interpolation)
        if (len(imglist) == 2) and self.resize_mask:
            imglist[1] = TF.resize(imglist[1], self.size, Image.NEAREST)
        return imglist

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """RandomHorizontalFlip tranform redefinition
    """
    def __call__(self, imglist):
        assert len(imglist) <= 2
        if random.random() < self.p:
            return [TF.hflip(img) for img in imglist]
        return imglist

class RandomVerticalFlip(T.RandomVerticalFlip):
    """RandomVerticalFlip transform redefinition
    """
    def __call__(self, imglist):
        assert len(imglist) <= 2
        if random.random() < self.p:
            return [TF.vflip(img) for img in imglist]
        return imglist

class ToTensor:
    """ToTensor transform redefinition
    """
    def __call__(self, imglist):
        assert len(imglist) <= 2
        return [TF.to_tensor(img) for img in imglist]

class Normalize(T.Normalize):
    """Normalize transform redefinition
    """
    def __call__(self, tensorlist):
        assert len(tensorlist) <= 2
        tensorlist[0] = TF.normalize(tensorlist[0], self.mean, self.std, self.inplace)
        return tensorlist

    def inverse(self, tensor):
        """Applies inverse transformation on specified tensor
            tensor: normalised tensor (C, H, W)
        Returns denormalized tensor
        """
        std = 1 / np.asarray(self.std)
        mean = - np.asarray(self.mean) * std
        std = tuple(std)
        mean = tuple(mean)
        return TF.normalize(tensor, mean, std)


## Dataset

class USNSDataset(Dataset):
    """USNS Dataset object
    """
    def __init__(self,
                 datadir,
                 img_list=None,
                 train=True,
                 transform=T.ToTensor(),
                 quiet=False):
        """
        Dataset class for loading images and masks.
        Data as in https://www.kaggle.com/c/ultrasound-nerve-segmentation/data
            datadir (string): Path to images
            img_list: image filenames to include in dataset. If None, includes
                all images from datadir. This list should not contain masks.
            train (bool): if True, read also a ground truth mask. Otherwise,
                mask filename
            transform (callable): transform to be applied on a
                sample
        """
        self.datadir = datadir
        self.train = train
        self.transform = transform

        if img_list is None:
            self.imglist = [ii for ii in os.listdir(datadir) if 'mask' not in ii]
        else:
            self.imglist = img_list

        self.imglist.sort(
            key=lambda fname: tuple(map(int, fname.split('.')[0].split('_')))
        )

        if self.train and not quiet:
            num_nerves = 0
            for img_fname in self.imglist:
                mask_fname = img_fname.replace('.tif', '_mask.tif')
                mask = Image.open(os.path.join(self.datadir, mask_fname))
                if mask.getbbox():
                    num_nerves += 1
                mask.close()
            empty_portion = 1 - num_nerves / len(self.imglist)
            print(f"The set contains {len(self.imglist)} images. Portion of empty masks: {empty_portion}")


    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        img_fname = self.imglist[index]
        mask_fname = img_fname.replace('.tif', '_mask.tif')

        img = Image.open(os.path.join(self.datadir, img_fname))
        if self.train:
            mask = Image.open(os.path.join(self.datadir, mask_fname))
        else:
            mask = mask_fname

        if self.transform:
            if self.train:
                img, mask = self.transform([img, mask])
            else:
                img = self.transform([img])[0]

        return img, mask


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
        self.loss = cross_entropy_loss

        self.n_samples = []
        self.loss_history = []
        self.val_loss_history = []
        self.dice_history = []

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_model_dice = 0.0

    def fit(self, dataloader, dataloader_val,
            epochs=1, print_every=1000, lr=0.001, loss_type='cross_entropy'):
        """
        Train model.
        Args:
            dataloader: pytorch compatible DataLoader, that provides
                minibatches of input preprocessed images and corresponding
                labled target images
            dataloader_val: pytorch compatible DataLoader for validation
            epochs: number of epochs to train
            print_every: num of samples to be processed to check val score
            loss_type: string, type of loss function from self.losses
            params: dict with training hyperparameters
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.n_samples = []
        self.loss_history = []
        self.val_loss_history = []
        self.dice_history = []

        if loss_type == 'cross_entropy':
            self.loss = cross_entropy_loss
        elif loss_type == 'dice':
            self.loss = lambda s, t: dice_loss(s, t, self.eps_dice)
        print(f"Optimization with {loss_type} loss function")

        running_loss = 0
        num_images = 0
        num_images_previous = 0
        for ee in range(epochs):
            for x, y in dataloader:
                self.model.train()
                x = x.to(self.device)
                y = y.to(self.device)
                y = y.squeeze(1).long()

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
        print(f'Best model dice: {self.best_model_dice}')

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
                y = y[:, 0, :, :].long()

                batch_size = y.shape[0]
                scores = self.model(x)
                loss += batch_size * self.loss(scores, y).item()

                pred = torch.argmax(scores, dim=1)
                # pred = F.pad(pred, (2, 2, 2, 2))
                dice += torch.sum(self.dice(pred, y))

                n_processed += batch_size
            loss = loss / n_processed
            dice = dice / n_processed

            if printing:
                print(f'Validation loss = {loss}')
                print(f'Val Dice score = {dice}')

            if show:
                show_composed_imgs(
                    x.cpu(),
                    y.cpu().unsqueeze(1),
                    pred.float().cpu().unsqueeze(1)
                )

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
        Returns tensor with predicted mask images
        """
        self.model.eval()
        with torch.no_grad():
            scores = self.model(x)
            preds = torch.argmax(scores, dim=1)
            # preds = F.pad(preds, (2, 2, 2, 2))
            preds = preds.float()
            preds = preds.cpu()
            preds = resize_tensor(preds.unsqueeze(1), (420, 580))
            preds = preds.to(self.device)

        return preds


def resize_tensor(tensor, size):
    """
    Rescale images in 4D tensor to desired size.
        tensor (pytorch tensor): input tensor with batch of images
        size (tupple): desired size
    """
    batch_size, channels, _, _ = tensor.shape
    res = torch.zeros(batch_size, channels, size[0], size[1], dtype=tensor.dtype)

    for ii in range(batch_size):
        img = TF.to_pil_image(tensor[ii])
        img = TF.resize(img, size, interpolation=Image.NEAREST)
        res[ii] = TF.to_tensor(img)

    return res

def cross_entropy_loss(scores, target):
    """
    Loss function used for base model trainig
    args:
        scores: model output scores
        target: ground truth lables
    returns:
        scalar loss for input minibatch
    """
    #scores = F.pad(scores, (2, 2, 2, 2))
    loss = F.cross_entropy(scores, target)

    return loss

def dice_loss(scores, target, eps=1e-6):
    """
    Dice loss function. Should be minimized! Accepts raw scores without
    softmax or sigmoid.
    args:
        scores: model output scores
        target:
    """
    scores = scores.flatten(start_dim=2)
    scores = F.softmax(scores, dim=1)
    target = target.flatten(start_dim=1)

    Dpos = (torch.sum(scores[:, 1, :] * target, dim=-1) + eps) / \
        (0.5*torch.sum(scores[:, 1, :] + target, dim=-1) + eps)
    Dneg = (torch.sum(scores[:, 0, :] * (1 - target), dim=-1) + eps) / \
        (0.5*torch.sum(scores[:, 0, :] + 1 - target, dim=-1) + eps)

    loss = 1 - 0.5*(Dpos + Dneg)

    return loss.mean()

def compose_visualization(snapshots, targets, predictions, alpha=0.7,
                          transform=Normalize((0.38983212684516944,), (0.21706658034222048,))):
    """Makes an image from snapshots with marked ground truth targets and
    predictions
        snapshots: tensor made from input images
        targets: tensor with relevant targets
        predictions: tensor with predictions
    """
    batch_size = snapshots.shape[0]
    imglist = []
    for n in range(batch_size):
        snap = transform.inverse(snapshots[n].cpu())
        snap = snap[0].numpy()
        targ = targets[n, 0].cpu().numpy()
        pred = predictions[n, 0].cpu().numpy()

        composed = np.stack([snap, snap, snap], axis=2)
        masks = np.stack([pred*(1-targ), targ*pred, targ*(1-pred)], axis=2)
        mask_idxs = masks.sum(axis=2) > 0
        composed[mask_idxs] = \
            (1 - alpha) * composed[mask_idxs] + alpha * masks[mask_idxs]
        imglist.append(composed)

    return imglist

def show_composed_imgs(snap, targ, pred):
    """Show predictions and targets above input image using composition
    by compose_visualization function
        input: model input image
        targ: ground truth segmentation
        pred: prediction segmentation
    """
    cols = 4

    imglist = compose_visualization(snap, targ, pred)
    batch_size = len(imglist)
    rows = np.int(np.ceil(batch_size / cols))

    _, axarr = plt.subplots(rows, cols, figsize=(4*cols, 4 * rows), squeeze=False)
    with torch.no_grad():
        for ii, img in enumerate(imglist):
            row = ii // cols
            col = ii % cols
            axarr[row, col].axis('off')
            axarr[row, col].imshow(img)
        plt.show()


def show_imgs(*imgs):
    """Show images from list of tensors. Grid of the size
    (batch_size x len(imgs)) is shown
    """
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
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def train_val_split(datadir, **options):
    """
    Splits images in datadir to training and validation set of patients.
        datadir: path to directory with data
        options: parameters passed to scikit-learn's train_test_split
    Returns two lists of images: training and validation sets
    """
    imglist = [ii for ii in os.listdir(datadir) if 'mask' not in ii]
    patients = set()
    for fname in imglist:
        patients.add(int(fname.split('_')[0]))
    patients = list(patients)
    patients.sort()

    patients_train, patients_val = train_test_split(patients, **options)
    imglist_train = [fname for fname in imglist if int(fname.split('_')[0]) in patients_train]
    imglist_val = [fname for fname in imglist if int(fname.split('_')[0]) in patients_val]

    return imglist_train, imglist_val


def get_labeled(datadir):
    """Select only labled images from datadir.
        datadir: path to dir with images
    Returns: list of images that has non-empty mask
    """
    labeled = []
    for imgname in os.listdir(datadir):
        if 'mask' not in imgname:
            maskname = imgname.replace('.tif', '_mask.tif')
            mask = Image.open(os.path.join(datadir, maskname))
            if np.array(mask).any():
                labeled.append(imgname)
    return labeled
