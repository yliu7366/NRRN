from .denoise_net import N2N_Net
from .nrrn import NRRN
from .unet import *

import torch.nn as nn

def get_net(input_depth, NET_TYPE):
    if NET_TYPE =='UNet':
        net=N2N_Unet()
    elif NET_TYPE == 'NRRN':
        net=NRRN(input_depth)
    elif NET_TYPE == "Denoise_Net":
        net=N2N_Net()
    else:
        assert False

    return net
