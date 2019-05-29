"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()

        return h, None

class CategoricalVideoDiscriminator(VideoDiscriminator):
    def __init__(self, n_channels, dim_categorical, n_output_neurons=1, use_noise=False, noise_sigma=None):
        super(CategoricalVideoDiscriminator, self).__init__(n_channels=n_channels,
                                                            n_output_neurons=n_output_neurons + dim_categorical,
                                                            use_noise=use_noise,
                                                            noise_sigma=noise_sigma)

        self.dim_categorical = dim_categorical

    def split(self, input):
        return input[:, :input.size(1) - self.dim_categorical], input[:, input.size(1) - self.dim_categorical:]

    def forward(self, input):
        h, _ = super(CategoricalVideoDiscriminator, self).forward(input)
        labels, categ = self.split(h)
        return labels, categ
    
    
class VBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, downsample_1dim=True, spectral_normalization=True):
        super(VBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample
        self.downsample_1dim = downsample_1dim

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        #self.c1 = utils.spectral_norm(nn.Conv3d(in_ch, h_ch, ksize, 1, pad))
        self.c1 = nn.Conv3d(in_ch, h_ch, ksize, stride=(1, 1, 1), padding=(1, 1, 1))
        #self.c2 = utils.spectral_norm(nn.Conv3d(h_ch, out_ch, ksize, 1, pad))
        self.c2 = nn.Conv3d(h_ch, out_ch, ksize, stride=(1, 1, 1), padding=(1, 1, 1))
        
        if self.learnable_sc:
            #self.c_sc = utils.spectral_norm(nn.Conv3d(in_ch, out_ch, 1, 1, 0))
            self.c_sc = nn.Conv3d(in_ch, out_ch, 1, stride=(1, 1, 1), padding=(0, 0, 0))

        if spectral_normalization:
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)
            self.c_sc = utils.spectral_norm(self.c_sc)
            
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        if self.downsample_1dim:
            kernel_size = (2, 2, 2)
        else:
            kernel_size = (1, 2, 2)
        return self.shortcut(x, kernel_size) + self.residual(x, kernel_size)

    def shortcut(self, x, kernel_size):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool3d(x, kernel_size=kernel_size)
        return x

    def residual(self, x, kernel_size):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool3d(h, kernel_size=kernel_size)
        return h


class OptimizedVBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu, spectral_normalization=True):
        super(OptimizedVBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv3d(in_ch, out_ch, ksize, stride=(1, 1, 1), padding=(1, 1, 1))
        self.c2 = nn.Conv3d(out_ch, out_ch, ksize, stride=(1, 1, 1), padding=(1, 1, 1))
        self.c_sc = nn.Conv3d(in_ch, out_ch, 1, 1, 0)

        if spectral_normalization:
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)
            self.c_sc = utils.spectral_norm(self.c_sc)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool3d(x, kernel_size=(1, 2, 2)))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool3d(self.c2(h), kernel_size=(1, 2, 2))
    
    
class SNResNetProjectionVideoDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes=0, activation=F.relu, spectral_normalization=True):
        super(SNResNetProjectionVideoDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedVBlock(3, num_features, 
                                      spectral_normalization=spectral_normalization)
        self.block2 = VBlock(num_features, num_features * 2,
                            activation=activation, downsample=True, downsample_1dim=False, 
                             spectral_normalization=spectral_normalization)
        self.block3 = VBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block4 = VBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block5 = VBlock(num_features * 8, num_features * 16,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block6 = VBlock(num_features * 16, num_features * 16,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.l7 = nn.Linear(num_features * 16, 1)
        if num_classes > 0:
            self.l_y = nn.Embedding(num_classes, num_features * 16)
            if spectral_normalization:
                self.l_y = utils.spectral_norm(self.l_y)
            
        if spectral_normalization:
            self.l7 = utils.spectral_norm(self.l7)
        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3)).squeeze()
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output
    
    
from torch.nn import init
from torch.nn import utils


class DBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, spectral_normalization=True):
        super(DBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

        if spectral_normalization:
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)
            self.c_sc = utils.spectral_norm(self.c_sc)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu, spectral_normalization=True):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_ch, out_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        
        if spectral_normalization:
            self.c1 = utils.spectral_norm(self.c1)
            self.c2 = utils.spectral_norm(self.c2)
            self.c_sc = utils.spectral_norm(self.c_sc)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)
    
    
class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features=32, num_classes=0, activation=F.relu, spectral_normalization=True):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features, spectral_normalization=spectral_normalization)
        self.block2 = DBlock(num_features, num_features * 2,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block3 = DBlock(num_features * 2, num_features * 4,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block4 = DBlock(num_features * 4, num_features * 8,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block5 = DBlock(num_features * 8, num_features * 16,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.block6 = DBlock(num_features * 16, num_features * 16,
                            activation=activation, downsample=True, spectral_normalization=spectral_normalization)
        self.l7 = nn.Linear(num_features * 16, 1)
        if num_classes > 0:
            self.l_y = nn.Embedding(num_classes, num_features * 16)
            if spectral_normalization:
                self.l_y = utils.spectral_norm(self.l_y)

        if spectral_normalization:
            self.l7 = utils.spectral_norm(self.l7)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output
    
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)
    
def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class GBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(GBlock, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))

class ResNetGenerator(nn.Module):
    """Generator generates 64x64."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 16 * num_features * bottom_width ** 2)

        self.block2 = GBlock(num_features * 16, num_features * 8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = GBlock(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = GBlock(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block5 = GBlock(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b6 = nn.BatchNorm2d(num_features)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z.squeeze()).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        
        y = y.long()
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b6(h))
        return torch.tanh(self.conv6(h))
                
class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=32, use_cgan_proj_discr=False, n_categories=None, resnet=True):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        self.use_cgan_proj_discr = use_cgan_proj_discr
        self.n_categories = n_categories
        self.resnet = True

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        if self.use_cgan_proj_discr or self.resnet:
            self.main = ResNetGenerator(num_features=32, dim_z=dim_z, num_classes=n_categories, bottom_width=4)
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples, video_len):
        video_len = video_len if video_len is not None else self.video_length

        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)

        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)

    def sample_z_video(self, num_samples, video_len=None):
        z_content = self.sample_z_content(num_samples, video_len)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        if z_category is not None:
            z = torch.cat([z_content, z_category, z_motion], dim=1)
        else:
            z = torch.cat([z_content, z_motion], dim=1)

        return z, z_category_labels

    def one_hot_v(self, data, num_classes):
        ones = torch.sparse.torch.eye(num_classes)
        return ones.index_select(0, data)
    
    def one_hot_expand(self, data, num_classes, len_expand):
        ones = self.one_hot_v(data, num_classes)
        return ones.expand(len_expand, ones.shape[0], ones.shape[1]).transpose(0, 1).reshape(-1, num_classes)
    
    def sample_videos(self, num_samples, video_len=None, use_cgan_proj_discr=False):
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels = self.sample_z_video(num_samples, video_len)
        z_category_labels = torch.from_numpy(z_category_labels)
        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()
        #z_category_labels = self.one_hot_expand(z_category_labels, self.n_categories, video_len)
        z_category_labels_exp = z_category_labels.expand(video_len, z_category_labels.shape[0]).t().flatten()

        #h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        if not use_cgan_proj_discr:
            h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        else:
            h = self.main(z.view(z.size(0), z.size(1), 1, 1), z_category_labels_exp)
            
        h = h.view(h.size(0) // video_len, video_len, self.n_channels, h.size(3), h.size(3))

        h = h.permute(0, 2, 1, 3, 4)
        return h, z_category_labels

    def sample_images(self, num_samples, use_cgan_proj_discr=False):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        z_category_labels = torch.from_numpy(z_category_labels)
        #z_category_labels = self.one_hot_expand(z_category_labels, self.n_categories, self.video_length)
        z_category_labels = z_category_labels.expand(self.video_length, z_category_labels.shape[0]).t().flatten()
        
        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        #z_category_labels = z_category_labels[j, ::]
        z_category_labels = z_category_labels[j]
        z = z.view(z.size(0), z.size(1), 1, 1)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()
            
        if not use_cgan_proj_discr:
            h = self.main(z)
            
            return h, None
        else:
            h = self.main(z, z_category_labels)
            
            return h, z_category_labels.long()


    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

