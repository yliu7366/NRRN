import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from .unet_parts import UNet

class N2C_Unet(nn.Module):
    """
    Unet applied to average of 3 images in a noise2clean training
    """
    def __init__(self):
        super(N2C_Unet, self).__init__()
        self.dn2n = UNet(1,1,False)
    def forward(self, *x):
        input1,input2,input3=tuple(x)
        x1 = input1
        x2 = ((input1 + input2 + input3)/3.)

        y = self.dn2n(x2)
        return y

class N2N_Unet(nn.Module):
    """
    Unet applied to 3 images in a noise2noise training
    """
    def __init__(self):
        super(N2N_Unet, self).__init__()
        self.dn2n = UNet(1,1,False)

    def forward(self, x):
        return self.dn2n(x)

