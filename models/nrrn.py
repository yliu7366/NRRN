import torch
import torch.nn as nn
import torch.nn.init as init

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class BuildingUnit(nn.Module):
    """
    Generate the Bulding Unit
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gate_update_a = nn.Conv2d(input_size,hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Gate_update_b = nn.Conv2d(hidden_size, hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Gate_reset_a = nn.Conv2d(input_size, hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Gate_reset_b = nn.Conv2d(hidden_size, hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Gates2_a = nn.Conv2d(input_size,hidden_size, KERNEL_SIZE, padding=PADDING)
        self.Gates2_b = nn.Conv2d(hidden_size, hidden_size, KERNEL_SIZE, padding=PADDING)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self._initialize_weights()

    def forward(self, input1,input2):
        # get batch and spatial sizes
        batch_size = input2.data.size()[0]
        spatial_size = input2.data.size()[2:]

        prev_cell = self.tanh(input1)
        # data size is [batch, channel, height, width]
        reset_gate = self.Gate_reset_a(prev_cell) + self.Gate_reset_b(input2)
        update_gate = self.Gate_update_a(prev_cell) + self.Gate_update_b(input2)

        # apply sigmoid non linearity
        reset_gate = self.sigmoid(reset_gate)
        update_gate = self.sigmoid(update_gate)

        cell_gate = self.Gates2_b(input2) + reset_gate*self.Gates2_a(prev_cell)
        # compute hidden state
        hidden = update_gate*prev_cell + (1 - update_gate)*self.tanh(cell_gate)

        return hidden, prev_cell

    def _initialize_weights(self):
        init.orthogonal_(self.Gate_update_a.weight, init.calculate_gain('sigmoid'))
        init.orthogonal_(self.Gate_update_b.weight, init.calculate_gain('sigmoid'))
        init.orthogonal_(self.Gate_reset_a.weight, init.calculate_gain('sigmoid'))
        init.orthogonal_(self.Gate_reset_b.weight, init.calculate_gain('sigmoid'))
        init.orthogonal_(self.Gates2_a.weight, init.calculate_gain('tanh'))
        init.orthogonal_(self.Gates2_b.weight, init.calculate_gain('tanh'))


def feedback(x,y):
    ##x--memory batch,64,256,256 ; y--updated_image batch,1,256,256
    batch_size = x.data.size()[0]
    spatial_size1, spatial_size2 = x.data.size()[2:]
    channel_size = x.data.size()[1]

    o = torch.ones(y.shape, device=y.device)
    z = torch.zeros((batch_size, channel_size - 1, spatial_size1, spatial_size2), device=x.device)

    mask = torch.cat([o, z], 1)
    h = x
    h = h*(1 - mask) + mask*torch.tanh(y)

    return h


class NRRN_Block(nn.Module):
    def __init__(self, bu_count=5, hidden_size=64):
        super(NRRN_Block,self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, hidden_size, 3, 1, 1)
        self.hidden_size = hidden_size
        self.bu_count = bu_count
        self.BU = nn.ModuleList()
        for i in range(bu_count):
            self.BU.append(BuildingUnit(hidden_size,hidden_size))

        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 3, 1, 1)
        self.noise_comp = NAtt()

    def forward(self, input):

        x0, x1 = torch.split(input, 1, 1)#input.chunk(2,1)
        num_batch = x1.shape[0]
        x_updated = x1.clone()


        h = self.relu(self.conv1(torch.cat((x0,x1), 0)))
        memory, new_inp = torch.split(h, num_batch, 0)

        # Building Unit
        for i, bu in enumerate(self.BU):
            memory = self.conv3(memory)
            hidden,memory = bu(memory, new_inp)
            x_updated += self.noise_comp(hidden)
            new_inp = feedback(hidden, x_updated)

        # The last  Conv Layers
        new_inp = self.conv2(self.relu(new_inp.clone())) # batch,63,256,256 ->batch,64
        x_updated += self.noise_comp(new_inp)

        return x_updated

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.max(x,1)[0].unsqueeze(1)


class NAtt(nn.Module):
    def __init__(self):
        super(NAtt, self).__init__()
        self.compress = ChannelPool()

    def forward(self, x):
        x_compress = self.compress(x)
        scale = torch.sigmoid(x_compress) # broadcasting
        x_chunks=torch.split(x, 1, dim=1)

        return x_chunks[0]*scale


class NRRN(nn.Module):
    def __init__(self, bu_count=1):
        super(NRRN, self).__init__()
        self.nrrn = NRRN_Block(bu_count)

    def forward(self, x):
        pred_x = self.nrrn(x)

        return pred_x
