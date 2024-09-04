import torch
import torch.nn.functional as F
from torch import nn



class Contrast(nn.Module):
    def __init__(self, num_convs=2, in_channels=256):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(Contrast, self).__init__()
        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        # initialization
        for modules in [self.dis_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)



    def forward(self, features_s, features_t):
        x_s = self.dis_tower(features_s)
        x_t = self.dis_tower(features_t)

        return x_s, x_t
