import os
import csv
import random
import argparse
from math import log10
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models
import n2n_loss
from data import get_training_set, get_test_set


parser = argparse.ArgumentParser(description='PyTorch DenoiseNet Example')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=50, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument("--data_dir", default="Data", type=str, help="Path to training data(default: Data/random_label)")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restart)")
parser.add_argument("--dataset", default="OHSU", type=str, help="dataset could be: EPFL,OHSU (default),TEST,OTH ")
parser.add_argument("--net", default="NRRN", type=str, help="net could be: NRRN(default) | Denoise_Net | UNet")

opt = parser.parse_args()
print(opt)

if opt.seed is not None:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
# LOADING the DATA
f_dir = opt.data_dir
if opt.dataset == "OHSU":         # OHSU FIB-SEM dataset
    train_set = get_training_set(f_dir)
    test_set = get_test_set(f_dir)
elif opt.dataset == "TEST":
    train_set = [torch.ones(10,1,256,256),torch.ones(10,1,256,256),torch.ones(10,1,256,256),torch.ones(10,1,256,256)]
    test_set = [torch.ones(10,1,256,256),torch.ones(10,1,256,256),torch.ones(10,1,256,256),torch.ones(10,1,256,256)]
else:                           # for EPFL dataset with added synthesized noise
    train_set = get_training_set(f_dir,synthesize=True)
    test_set = get_test_set(f_dir,synthesize=True)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
print("training set size:",len(train_set))


print('===> Building model')
# Defining the Model options are NRRN, UNet Denois_Net
# input_depth = number of Building Units
model = models.get_net(input_depth=1, NET_TYPE=opt.net)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
        print("Checkpoint's state_dict:")
        model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
else:
    opt.start_epoch = 1

model = nn.DataParallel(model)
model.to(device)
criterion1 = nn.MSELoss().to(device)
criterion2 = n2n_loss.N2NLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#print(model)

print("===> Start Training")


def train(epoch):
    epoch_loss = 0
    psnr = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        optimizer.zero_grad()

        input0, input1, input2 = batch[0].to(device), batch[1].to(device),batch[2].to(device)
        target = batch[3].to(device)
        # concat the 2 images on the channel dim => (batch,2,256,256)
        if opt.net == 'NRRN':
            x = torch.cat((input1,input0),dim=1)
            y = torch.cat((input2,input0),dim=1)
        elif opt.net == 'Denoise_Net':
            x = torch.cat((input0,input1),dim=1)
            y = torch.cat((input0,input2),dim=1)
        elif opt.net == 'UNet':
            x = (input1+input0)/2.
            y = (input2+input0)/2.

        out_x = model(x)
        out_y = model(y)

        loss = criterion2(input1, input2, out_x, out_y)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        mse = criterion1((out_x + out_y)/2., target)
        psnr += 10*log10(1/(mse.item() + 1e-8))

    print("===> Epoch {} Complete: Avg. Loss: {:.5f} PSNR: {:.5f}".format(epoch,
        epoch_loss/len(training_data_loader),psnr/len(training_data_loader)))

    return ( epoch_loss / len(training_data_loader),
                    psnr/len(training_data_loader))


def test():
    avg_psnr = 0
    epoch_loss = 0
    with torch.no_grad():
        for itr,batch in enumerate(testing_data_loader,1):
            input0, input1, input2 = batch[0].to(device), batch[1].to(device),batch[2].to(device)
            target = batch[3].to(device)
            # concat the 2 images on the channel dim => (batch,2,256,256)
            if opt.net == 'NRRN':
                x = torch.cat((input1,input0),dim=1)
                y = torch.cat((input2,input0),dim=1)
            elif opt.net == 'Denoise_Net':
                x = torch.cat((input0,input1),dim=1)
                y = torch.cat((input0,input2),dim=1)
            elif opt.net == 'UNet':
                x = (input1 + input0)/2.
                y = (input2 + input0)/2.

            out_x = model(x)
            out_y = model(y)
            prediction = (out_y + out_x)/2.

            loss = criterion2(input1, input2, out_x, out_y)
            epoch_loss += loss.item()

            mse = criterion1(prediction, target)
            psnr = 10*log10(1/mse.item())
            avg_psnr += psnr

    print("===> Val Avg. PSNR: {:.4f} dBi, Avg Val Loss: {:.5f}".format(avg_psnr
        /len(testing_data_loader),epoch_loss/len(testing_data_loader)))

    return (epoch_loss/len(testing_data_loader), avg_psnr /len(testing_data_loader))


def checkpoint(epoch,name="model_epoch"):
    """
    Saving the model
    """
    model_out_path = name+"_{}.pth".format(epoch)
    torch.save(model.module.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###------MAIN CODE -------------------------------------------------###
name_string = 'Loss_bS'+str(opt.batchSize)+'tBS'+str(opt.testBatchSize)+'E'+str(opt.nEpochs)+'lr'+str(int(log10(opt.lr)))+'.csv'
min_ltest = 1.e3          # a large number
min_ltest_epoch = 0       # before training
with open(name_string,mode = 'w') as f:
    # Saving the Epoch, train loss/PSNR values and test loss/PSNR values
    f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    f_writer.writerow([ "Epoch", "Loss_Train", "Loss_Test", "TrainPSNR", "TestPSNR" ])

    ltrain = 0.0
    ltest = 0.0
    psnr_test = 0.0
    psnr_train = 0.0

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        ltrain,psnr_train = train(epoch)
        ltest,psnr_test = test()
        scheduler.step()

        if (ltest < min_ltest):
            min_ltest = ltest
            min_ltest_epoch = epoch
            checkpoint(epoch,"BestModel_epoch")

        f_writer.writerow([epoch,ltrain,ltest,psnr_train, psnr_test])
