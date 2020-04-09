import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    """Swish activation function by Google
    $Swish = x * \sigma(x)$
    """
    return x * torch.sigmoid(x)

activations = {
            'relu': F.relu,
            'swish': swish
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pad=0, bn=False, activation='relu'):
        """
        Convolutional block of U-net architecture without final activation
        (it is optimal to make ReLU after max pool)
        """
        super().__init__()
        self.bn = bn
        self.activation = activations[activation]

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               (3, 3), padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               (3, 3), padding=pad, bias=True)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn: x = self.bn1(x)
        x = self.conv2(self.activation(x))
        if self.bn: x = self.bn2(x)

        return x


class UpPool(nn.Module):
    """
    Up convolution on the way up
    Accepts input x from previouse layer and concatenates output with
    features f from down pass
    """
    def __init__(self, in_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channel, in_channel // 2,
                                         (2, 2), stride=2, bias=True)
    
    def forward(self, x, f):
        x = self.upconv(F.relu(x))
        # do we need relu for x here?
        out = F.relu(torch.cat([f, x], dim=1))

        return out


class Swish(nn.Module):
    """Swish activation function by Google
    $Swish = x * \sigma(x)$
    """
    def forward(self, x):
        return swish(x)

class Unet(nn.Module):
    def __init__(self, n_filters=64, bn=False):
        """
        Unet CNN for Ultrasound Nerve Segmentation challenge
            n_filters: (int) initial number of filters
            bn: (bool) if use of batch normalization
        """
        super().__init__()
        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
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
        return F.relu(F.max_pool2d(x, (2, 2)))

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
        x = self.outconv(F.relu(x))

        return x


class Unet3(nn.Module):
    def __init__(self, n_filters=64, bn=False):
        super().__init__()
        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
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
        return F.relu(F.max_pool2d(x, (2, 2)))

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
        x = self.outconv(F.relu(x))

        return x


class Unet1(nn.Module):
    """
    Unet CNN for Ultrasound Nerve Segmentation challenge
    Depth 1.
        n_filters: (int) initial number of filters
        bn: (bool) if use of batch normalization
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()
        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        # bottom
        self.bottom = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        # up
        self.upconv1 = UpPool(2*n_filters)
        self.up_blk1 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)
        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)
    
    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))

        # bottom
        x = self.bottom(x)

        # up
        x = self.up_blk1(self.upconv1(x, out1))

        x = self.outconv(F.relu(x))

        return x


class Unet2(nn.Module):
    """
    Unet CNN for Ultrasound Nerve Segmentation challenge
    Depth 2.
        n_filters: (int) initial number of filters
        bn: (bool) if use of batch normalization
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)

        # bottom
        self.bottom = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(4*n_filters)
        self.up_blk1 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(2*n_filters)
        self.up_blk2 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)
        

        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)
    
    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))
        out2 = self.dn_blk2(x)
        x = F.relu(F.max_pool2d(out2, (2, 2)))

        # bottom
        x = self.bottom(x)

        # up
        x = self.up_blk1(self.upconv1(x, out2))
        x = self.up_blk2(self.upconv2(x, out1))

        x = self.outconv(F.relu(x))

        return x


class Unet5(nn.Module):
    """
    Unet CNN for Ultrasound Nerve Segmentation challenge
    Depth 5.
        n_filters: (int) initial number of filters
        bn: (bool) if use of batch normalization
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        self.dn_blk5 = ConvBlock(8*n_filters, 16*n_filters, pad=1, bn=bn)

        # bottom
        self.bottom = ConvBlock(16*n_filters, 32*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(32*n_filters)
        self.up_blk1 = ConvBlock(32*n_filters, 16*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(16*n_filters)
        self.up_blk2 = ConvBlock(16*n_filters, 8*n_filters, pad=1, bn=bn)
        self.upconv3 = UpPool(8*n_filters)
        self.up_blk3 = ConvBlock(8*n_filters, 4*n_filters, pad=1, bn=bn)
        self.upconv4 = UpPool(4*n_filters)
        self.up_blk4 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv5 = UpPool(2*n_filters)
        self.up_blk5 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)
        
        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)
    
    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))
        out2 = self.dn_blk2(x)
        x = F.relu(F.max_pool2d(out2, (2, 2)))
        out3 = self.dn_blk3(x)
        x = F.relu(F.max_pool2d(out3, (2, 2)))
        out4 = self.dn_blk4(x)
        x = F.relu(F.max_pool2d(out4, (2, 2)))
        out5 = self.dn_blk5(x)
        x = F.relu(F.max_pool2d(out5, (2, 2)))

        # bottom
        x = self.bottom(x)

        # up
        x = self.up_blk1(self.upconv1(x, out5))
        x = self.up_blk2(self.upconv2(x, out4))
        x = self.up_blk3(self.upconv3(x, out3))
        x = self.up_blk4(self.upconv4(x, out2))
        x = self.up_blk5(self.upconv5(x, out1))

        x = self.outconv(F.relu(x))

        return x


class Unet6(nn.Module):
    """
    Unet CNN for Ultrasound Nerve Segmentation challenge
    Depth 6.
        n_filters: (int) initial number of filters
        bn: (bool) if use of batch normalization
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        self.dn_blk5 = ConvBlock(8*n_filters, 16*n_filters, pad=1, bn=bn)
        self.dn_blk6 = ConvBlock(16*n_filters, 32*n_filters, pad=1, bn=bn)

        # bottom
        self.bottom = ConvBlock(32*n_filters, 64*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(64*n_filters)
        self.up_blk1 = ConvBlock(64*n_filters, 32*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(32*n_filters)
        self.up_blk2 = ConvBlock(32*n_filters, 16*n_filters, pad=1, bn=bn)
        self.upconv3 = UpPool(16*n_filters)
        self.up_blk3 = ConvBlock(16*n_filters, 8*n_filters, pad=1, bn=bn)
        self.upconv4 = UpPool(8*n_filters)
        self.up_blk4 = ConvBlock(8*n_filters, 4*n_filters, pad=1, bn=bn)
        self.upconv5 = UpPool(4*n_filters)
        self.up_blk5 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv6 = UpPool(2*n_filters)
        self.up_blk6 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)

        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)

    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))
        out2 = self.dn_blk2(x)
        x = F.relu(F.max_pool2d(out2, (2, 2)))
        out3 = self.dn_blk3(x)
        x = F.relu(F.max_pool2d(out3, (2, 2)))
        out4 = self.dn_blk4(x)
        x = F.relu(F.max_pool2d(out4, (2, 2)))
        out5 = self.dn_blk5(x)
        x = F.relu(F.max_pool2d(out5, (2, 2)))
        out6 = self.dn_blk6(x)
        x = F.relu(F.max_pool2d(out6, (2, 2)))

        # bottom
        x = self.bottom(x)

        # up
        x = self.up_blk1(self.upconv1(x, out6))
        x = self.up_blk2(self.upconv2(x, out5))
        x = self.up_blk3(self.upconv3(x, out4))
        x = self.up_blk4(self.upconv4(x, out3))
        x = self.up_blk5(self.upconv5(x, out2))
        x = self.up_blk6(self.upconv6(x, out1))

        x = self.outconv(F.relu(x))

        return x


class Unet7(nn.Module):
    """
    Unet CNN for Ultrasound Nerve Segmentation challenge
    Depth 7.
        n_filters: (int) initial number of filters
        bn: (bool) if use of batch normalization
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        self.dn_blk5 = ConvBlock(8*n_filters, 16*n_filters, pad=1, bn=bn)
        self.dn_blk6 = ConvBlock(16*n_filters, 32*n_filters, pad=1, bn=bn)
        self.dn_blk7 = ConvBlock(32*n_filters, 64*n_filters, pad=1, bn=bn)

        # bottom
        self.bottom = ConvBlock(64*n_filters, 128*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(128*n_filters)
        self.up_blk1 = ConvBlock(128*n_filters, 64*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(64*n_filters)
        self.up_blk2 = ConvBlock(64*n_filters, 32*n_filters, pad=1, bn=bn)
        self.upconv3 = UpPool(32*n_filters)
        self.up_blk3 = ConvBlock(32*n_filters, 16*n_filters, pad=1, bn=bn)
        self.upconv4 = UpPool(16*n_filters)
        self.up_blk4 = ConvBlock(16*n_filters, 8*n_filters, pad=1, bn=bn)
        self.upconv5 = UpPool(8*n_filters)
        self.up_blk5 = ConvBlock(8*n_filters, 4*n_filters, pad=1, bn=bn)
        self.upconv6 = UpPool(4*n_filters)
        self.up_blk6 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv7 = UpPool(2*n_filters)
        self.up_blk7 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)

        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)

    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))
        out2 = self.dn_blk2(x)
        x = F.relu(F.max_pool2d(out2, (2, 2)))
        out3 = self.dn_blk3(x)
        x = F.relu(F.max_pool2d(out3, (2, 2)))
        out4 = self.dn_blk4(x)
        x = F.relu(F.max_pool2d(out4, (2, 2)))
        out5 = self.dn_blk5(x)
        x = F.relu(F.max_pool2d(out5, (2, 2)))
        out6 = self.dn_blk6(x)
        x = F.relu(F.max_pool2d(out6, (2, 2)))
        out7 = self.dn_blk7(x)
        x = F.relu(F.max_pool2d(out7, (2, 2)))

        # bottom
        x = self.bottom(x)

        # up
        x = self.up_blk1(self.upconv1(x, out7))
        x = self.up_blk2(self.upconv2(x, out6))
        x = self.up_blk3(self.upconv3(x, out5))
        x = self.up_blk4(self.upconv4(x, out4))
        x = self.up_blk5(self.upconv5(x, out3))
        x = self.up_blk6(self.upconv6(x, out2))
        x = self.up_blk7(self.upconv7(x, out1))

        x = self.outconv(F.relu(x))

        return x


class Unet8(nn.Module):
    """
    Unet CNN for Ultrasound Nerve Segmentation challenge
    Depth 8. Max depth for 256x256 input image
        n_filters: (int) initial number of filters
        bn: (bool) if use of batch normalization
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        self.dn_blk5 = ConvBlock(8*n_filters, 16*n_filters, pad=1, bn=bn)
        self.dn_blk6 = ConvBlock(16*n_filters, 32*n_filters, pad=1, bn=bn)
        self.dn_blk7 = ConvBlock(32*n_filters, 64*n_filters, pad=1, bn=bn)
        self.dn_blk8 = ConvBlock(64*n_filters, 128*n_filters, pad=1, bn=bn)

        # bottom
        self.bottom = ConvBlock(128*n_filters, 256*n_filters, pad=1, bn=bn)

        # up
        self.upconv1 = UpPool(256*n_filters)
        self.up_blk1 = ConvBlock(256*n_filters, 128*n_filters, pad=1, bn=bn)
        self.upconv2 = UpPool(128*n_filters)
        self.up_blk2 = ConvBlock(128*n_filters, 64*n_filters, pad=1, bn=bn)
        self.upconv3 = UpPool(64*n_filters)
        self.up_blk3 = ConvBlock(64*n_filters, 32*n_filters, pad=1, bn=bn)
        self.upconv4 = UpPool(32*n_filters)
        self.up_blk4 = ConvBlock(32*n_filters, 16*n_filters, pad=1, bn=bn)
        self.upconv5 = UpPool(16*n_filters)
        self.up_blk5 = ConvBlock(16*n_filters, 8*n_filters, pad=1, bn=bn)
        self.upconv6 = UpPool(8*n_filters)
        self.up_blk6 = ConvBlock(8*n_filters, 4*n_filters, pad=1, bn=bn)
        self.upconv7 = UpPool(4*n_filters)
        self.up_blk7 = ConvBlock(4*n_filters, 2*n_filters, pad=1, bn=bn)
        self.upconv8 = UpPool(2*n_filters)
        self.up_blk8 = ConvBlock(2*n_filters, n_filters, pad=1, bn=bn)


        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)

    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))
        out2 = self.dn_blk2(x)
        x = F.relu(F.max_pool2d(out2, (2, 2)))
        out3 = self.dn_blk3(x)
        x = F.relu(F.max_pool2d(out3, (2, 2)))
        out4 = self.dn_blk4(x)
        x = F.relu(F.max_pool2d(out4, (2, 2)))
        out5 = self.dn_blk5(x)
        x = F.relu(F.max_pool2d(out5, (2, 2)))
        out6 = self.dn_blk6(x)
        x = F.relu(F.max_pool2d(out6, (2, 2)))
        out7 = self.dn_blk7(x)
        x = F.relu(F.max_pool2d(out7, (2, 2)))
        out8 = self.dn_blk8(x)
        x = F.relu(F.max_pool2d(out8, (2, 2)))

        # bottom
        x = self.bottom(x)

        # up
        x = self.up_blk1(self.upconv1(x, out8))
        x = self.up_blk2(self.upconv2(x, out7))
        x = self.up_blk3(self.upconv3(x, out6))
        x = self.up_blk4(self.upconv4(x, out5))
        x = self.up_blk5(self.upconv5(x, out4))
        x = self.up_blk6(self.upconv6(x, out3))
        x = self.up_blk7(self.upconv7(x, out2))
        x = self.up_blk8(self.upconv8(x, out1))

        x = self.outconv(F.relu(x))

        return x


class UnetD(nn.Module):
    """Unet with custom depth D
    """
    def __init__(self, depth, n_filters, bn=False, activation='relu'):
        super().__init__()
        self.depth = depth

        self.activation = activations[activation]

        # down
        self.dn_blks = nn.ModuleList()
        in_ch = 1
        out_ch = n_filters
        for dd in range(self.depth):
            self.dn_blks.append(ConvBlock(in_ch, out_ch, pad=1, bn=bn, activation=activation))
            in_ch = out_ch
            out_ch *= 2

        # bottom
        self.bottom = ConvBlock(in_ch, out_ch, pad=1, bn=bn, activation=activation)
        in_ch, out_ch = out_ch, in_ch

        # up
        self.upconvs = nn.ModuleList()
        self.up_blks = nn.ModuleList()
        for dd in range(self.depth):
            self.upconvs.append(UpPool(in_ch))
            self.up_blks.append(ConvBlock(in_ch, out_ch, pad=1, bn=bn, activation=activation))
            in_ch = out_ch
            out_ch = out_ch // 2

        # output
        self.outconv = nn.Conv2d(n_filters, 2, (1, 1), bias=True)

    def forward(self, x):
        outs = []
        for dn_blk in self.dn_blks:
            x = dn_blk(x)
            outs.append(x)
            x = self.activation(F.max_pool2d(x, (2, 2)))

        x = self.bottom(x)
        outs.reverse()

        for upconv, up_blk, out in zip(self.upconvs, self.up_blks, outs):
            x = up_blk(upconv(x, out))

        x = self.outconv(self.activation(x))

        return x


"""
Models for binary classification (is any nerve present on image)
"""

class BinNet1_left(nn.Module):
    """
    Binary classifier deciding if nerve present on image
    """
    def __init__(self, n_filters):
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(8*n_filters, 2*n_filters, kernel_size=(1, 1))
        self.conv1x1_2 = nn.Conv2d(2*n_filters, n_filters//2, kernel_size=(1, 1))
        self.fc = nn.Linear((n_filters//2)*3744, 1)

    def forward(self, x):
        x = F.relu(self.conv1x1_1(x))
        x = F.relu(self.conv1x1_2(x))
        x = self.fc(x.flatten(start_dim=1))

        return x


class Unet5Part(nn.Module):
    """Left part of Unet5 generating features to be used in nerve existence
    clasifier
    """
    def __init__(self, n_filters, bn=False):
        super().__init__()

        # down
        self.dn_blk1 = ConvBlock(1, n_filters, pad=1, bn=bn)
        self.dn_blk2 = ConvBlock(n_filters, 2*n_filters, pad=1, bn=bn)
        self.dn_blk3 = ConvBlock(2*n_filters, 4*n_filters, pad=1, bn=bn)
        self.dn_blk4 = ConvBlock(4*n_filters, 8*n_filters, pad=1, bn=bn)
        self.dn_blk5 = ConvBlock(8*n_filters, 16*n_filters, pad=1, bn=bn)

    def forward(self, x):
        # down
        out1 = self.dn_blk1(x)
        x = F.relu(F.max_pool2d(out1, (2, 2)))
        out2 = self.dn_blk2(x)
        x = F.relu(F.max_pool2d(out2, (2, 2)))
        out3 = self.dn_blk3(x)
        x = F.relu(F.max_pool2d(out3, (2, 2)))
        out4 = self.dn_blk4(x)
        x = F.relu(F.max_pool2d(out4, (2, 2)))
        out5 = self.dn_blk5(x)
        x = F.max_pool2d(out5, (2, 2))

        return x


class UnetDPart(nn.Module):
    """Left part of UnetD generating features to be used in nerve existence
    clasifier
    """
    def __init__(self, depth, n_filters, bn=False):
        super().__init__()
        self.depth = depth

        # down
        self.dn_blks = nn.ModuleList()
        in_ch = 1
        out_ch = n_filters
        for dd in range(self.depth):
            self.dn_blks.append(ConvBlock(in_ch, out_ch, pad=1, bn=bn))
            in_ch = out_ch
            out_ch *= 2

    def forward(self, x):
        for dn_blk in self.dn_blks[:-1]:
            x = dn_blk(x)
            x = F.relu(F.max_pool2d(x, (2, 2)))
        x = self.dn_blks[-1](x)
        x = F.max_pool2d(x, (2, 2))

        return x


class BinClf(nn.Module):
    """Binary classifier for bottom part of model output
    """
    def __init__(self, n_filters):
        super().__init__()
        self.conv8x8 = nn.Conv2d(16*n_filters, n_filters, (8, 8))
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, features):
        out = F.relu(self.conv8x8(features))
        out = out.flatten(start_dim=1)
        out = self.fc(out)

        return out
