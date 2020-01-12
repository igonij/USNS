import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0, bn=False):
        """
        Convolutional block of U-net architecture without activation (it is
        optimal to make ReLU after max pool)
        """
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.relu = nn.ReLU()

        if self.bn:
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)
    
    def forward(self, x):
        x = self.conv1(x)
        if self.bn: x = self.bn1(x)
        x = self.conv2(self.relu(x))
        if self.bn: x = self.bn2(x)
        
        return x


class UpPool(nn.Module):
    """
    Up convolution on the way up
    Acceprs input x from previouse layer and concatenates output with
    features f from down pass
    """
    def __init__(self, in_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channel, in_channel // 2,
                                         (2, 2), stride=2, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x, f):
        x = self.upconv(self.relu(x))
        # do we need relu for x here?
        out = self.relu(torch.cat([f, x], dim=1))

        return out 


class Unet(nn.Module):
    def __init__(self, n_filters=64, bn=False):
        """
        Unet CNN for Ultrasound Nerve Segmentation challenge
            n_filters: (int) initial number of filters
            bn: (bool)  yse of batch normalization
        """
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2))

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=0, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        
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

        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)

    def dnpool(self, x):
        return self.relu(self.maxpool(x))

    def forward(self, x):
        # go down
        out1 = self.dn_blk1(x)
        out2 = self.dn_blk2(self.dnpool(out1))
        out3 = self.dn_blk3(self.dnpool(out2))
        out4 = self.dn_blk4(self.dnpool(out3))
        
        # bottom block
        self.bottom_out = self.bottom_blk(self.dnpool(out4))
        
        # go up
        x = self.up_blk1(self.upconv1(self.bottom_out, out4))
        x = self.up_blk2(self.upconv2(x, out3))
        x = self.up_blk3(self.upconv3(x, out2))
        x = self.up_blk4(self.upconv4(x, out1))

        # out block
        x = self.outconv(x)

        return x


class Unet3(nn.Module):
    def __init__(self, n_filters=64, bn=False):
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 2))

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=0, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        
        # bottom
        self.bottom_blk = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(8*n_filters)
        self.up_blk1 = ConvBlock(8*n_filters, 4*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(4*n_filters)
        self.up_blk2 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv3 = UpPool(2*n_filters)
        self.up_blk3 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)
        
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)

    def dnpool(self, x):
        return self.relu(self.maxpool(x))

    def forward(self, x):
        # go down
        out1 = self.dn_blk1(x)
        out2 = self.dn_blk2(self.dnpool(out1))
        out3 = self.dn_blk3(self.dnpool(out2))
        
        # bottom block
        self.bottom_out = self.bottom_blk(self.dnpool(out3))
        
        # go up
        x = self.up_blk1(self.upconv1(self.bottom_out, out3))
        x = self.up_blk2(self.upconv2(x, out2))
        x = self.up_blk3(self.upconv3(x, out1))
        
        # out block
        x = self.outconv(x)

        return x