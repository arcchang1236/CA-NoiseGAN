import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
  def __init__(self, in_dim, out_dim, k=1, s=1, p=0, d=1, pad_type='zero', norm='none', sn=True, activation='leaky_relu', deconv=False):
    super(ConvBlock, self).__init__()
    
    layers = []
    # Conv
    if deconv is True:
      if sn is True:
        layers += [nn.utils.spectral_norm(nn.ConvTranspose2d(in_dim, out_dim, k, s, padding=p, dilation=d, bias=False))]
      else:
        layers += [nn.ConvTranspose2d(in_dim, out_dim, k, s, padding=p, dilation=d, bias=False)]
    else:            
      if sn is True:
        layers += [nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, k, s, padding=p, dilation=d, bias=False))]
      else:
        layers += [nn.Conv2d(in_dim, out_dim, k, s, padding=p, dilation=d, bias=False)]
    # Norm
    if norm == 'bn':
      layers += [nn.BatchNorm2d(out_dim, affine=True)]
    elif norm == 'inn':
      layers += [nn.InstanceNorm2d(out_dim, affine=True)]
    
    # Activation
    if activation == 'leaky_relu':
      layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
    elif activation == 'relu':
      layers += [nn.ReLU(inplace=True)]
    elif activation == 'sigmoid':
      layers += [nn.Sigmoid()]
    elif activation == 'tanh':
      layers += [nn.Tanh()]

    self.conv_block = nn.Sequential(*layers)
    
  def forward(self, x):
    out = self.conv_block(x)
    return out

class DeConvBlock(nn.Module):
  def __init__(self, in_dim, out_dim, k=1, s=1, p=0, d=1, pad_type='zero', norm='none', sn=True, activation='leaky_relu'):
    super(DeConvBlock, self).__init__()
    
    self.conv2d = ConvBlock(in_dim, out_dim, k=k, s=s, p=p, d=d, pad_type=pad_type, norm=norm, sn=sn, activation=activation, deconv=True)

  def forward(self, x):
    out = self.conv2d(x)
    return out

class ResnetBlock(nn.Module):
  def __init__(self, dim, k=3, s=1, p=1, d=1, pad_type='zero', norm='none', sn=True, activation='none', use_dropout=False):
    super(ResnetBlock, self).__init__()
    
    layers = [ConvBlock(dim, dim, k=k, s=s, p=p, d=d, pad_type=pad_type, norm=norm, sn=sn, activation=activation)]
    if use_dropout:
      layers += [nn.Dropout(p=0.5)]
    layers += [ConvBlock(dim, dim, k=k, s=s, p=p, d=d, pad_type=pad_type, norm='none', sn=sn, activation='none')]
    self.res_block = nn.Sequential(*layers)

  def forward(self, x):
    residual = x
    out = self.res_block(x)
    out = out + residual
    return out

class PyramidPooling(nn.Module):
  def __init__(self, in_channels, pool_sizes, norm, activation, pad_type='zero'):
    super(PyramidPooling, self).__init__()
    
    self.paths = []
    for i in range(len(pool_sizes)):
      self.paths.append(ConvBlock(in_channels, int(in_channels/len(pool_sizes)), k=1, s=1, p=0, pad_type=pad_type, norm=norm, activation=activation))
    self.path_module_list = nn.ModuleList(self.paths)
    self.pool_sizes = pool_sizes

  def forward(self, x):
    output_slices = [x]
    h, w = x.shape[2:]

    for module, pool_size in zip(self.path_module_list, self.pool_sizes): 
      out = F.avg_pool2d(x, int(h/pool_size), int(h/pool_size), 0)
      out = module(out)
      out = F.interpolate(out, size=(h,w), mode='bilinear')
      output_slices.append(out)

    return torch.cat(output_slices, dim=1)