import torch
import torch.nn as nn
import torch.nn.init as init


class BasicBlock(nn.Module):
    """
    Basic Block for the DenoiseNet
    """

    def __init__(self, bb=64):
        super(BasicBlock, self).__init__()
        self.bb = bb
        self.conv2 = nn.Conv2d(bb-1, bb,  3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(x))


class Net(nn.Module):
    """ DenoiseNet
        Parameters:
                    dim - defines how many images are going to be used for denoising e.g.1,3
                    count - the number of the layers, DenoiseNet uses 20
    """
    def __init__(self,bb_count=20,dim=2):
        super(Net, self).__init__()
        self.bb_count = bb_count
        self.BU = nn.ModuleList()
        for i in range(bb_count):
            self.BU.append(BasicBlock(64))


        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim,64, 3, 1, 1) # dim is used here

    def forward(self, x):
        input_x = x[:,0:1,:,:].clone()
        x = self.conv1(x)
        pred_noise = x[:,0:1,:,:].clone()

        for i, bu in enumerate(self.BU):
            x = bu(x[:,1:])
            pred_noise += x[:,0:1,:,:].clone()

        return input_x + pred_noise


class N2N_Net(nn.Module):
    """
        DenoiseNet used in a N2N setting
    """
    def __init__(self):
        super(N2N_Net, self).__init__()
        self.dn2n = Net(dim=2)

    def forward(self, x):
        y = self.dn2n(x)

        return y


class DenoiseNet(nn.Module):
    """
        The classical DenoiseNet model
    """
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.dn2n = Net(dim=3)

    def forward(self, *x):
        input1,input2,input3 = tuple(x)
        x1 = torch.cat((input1,input2,input3),dim=1)
        out = self.dn2n(x1)

        return out
