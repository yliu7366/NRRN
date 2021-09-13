import numpy as np
import torch
import torch.nn as nn

class N2NLoss(nn.MSELoss):
    """
    Implementation of the Noise2Noise Loss Function of Dufan Wu
    https://arxiv.org/pdf/1906.03639.pdf

    """
    def __init__(self):
        super(N2NLoss, self).__init__()

    def forward(self, *inputs):
        input1, input2, pred_input1, pred_input2 = tuple(inputs)
        loss1 = super(N2NLoss, self).forward(pred_input1, input2)
        loss2 = super(N2NLoss, self).forward(pred_input2, input1)
        loss3 = super(N2NLoss, self).forward(pred_input1, pred_input2)

        return 0.5*loss1 + 0.5*loss2 - 0.25*loss3
