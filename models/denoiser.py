import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DnCNN(nn.Module):
  def __init__(self, num_layers=9, num_features=64):
    super(DnCNN, self).__init__()
    layers = [nn.Sequential(nn.Conv2d(4, num_features, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True))]
    for i in range(num_layers - 2):
      layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(num_features),
                                  nn.ReLU(inplace=True)))
    layers.append(nn.Conv2d(num_features, 4, kernel_size=3, padding=1))
    self.layers = nn.Sequential(*layers)

    self._initialize_weights()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

  def forward(self, inputs):
    y = inputs
    residual = self.layers(y)
    return y - residual