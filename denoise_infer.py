import argparse
from math import log10
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor,Normalize,Compose

import patches
import models


parser = argparse.ArgumentParser(description='PyTorch NRRN Denoiseing Inference')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--input_image2', type=str, default='A', required=False, help='input image to use')
parser.add_argument('--input_image3', type=str, default='A', required=False, help='input image to use')
parser.add_argument('--target_image', type=str, required=True, help='target image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--overlap', type=int, default=20, help='Patches overlap ex 0,7,13,20, default 20')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()
print(opt)

def psnr(img_orig,img_out):
    ### Calculating PSNR of 2 images
    Max = 255
    img_orig = np.array(img_orig).astype(float)/Max
    img_out = np.array(img_out).astype(float)/Max

    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0/mse)
    return psnr

def Image_from_tensor(im,Max):
    mu = 0.0
    std = 1.0
    im_numpy = im.numpy().squeeze()
    im_numpy = (im_numpy.astype(float)*std+mu)*Max
    im_numpy = im_numpy.clip(0.0,Max)
    numpy_img = Image.fromarray(np.uint8(im_numpy),mode='L')
    return numpy_img

device = torch.device("cuda" if opt.cuda else "cpu")
print("Device: ",device)

# Checking how many images are supplied for denosing
number_of_inputs = 1
if opt.input_image2 != 'A':
    number_of_inputs += 1
if opt.input_image3 != 'A':
    number_of_inputs +=1

img_pad = 1

# Amount of overlap between extracted image crops
patch_overlap = opt.overlap
if patch_overlap == 13:
     patch_size_w = 256
elif patch_overlap == 7:
     patch_size_w = 128
elif patch_overlap == 20:
     patch_size_w = 256
else:
     patch_size_w = 256
patch_size_h = patch_size_w-patch_overlap

# Appending the input images into array
img_array = []
img_array.append(opt.input_image)
if number_of_inputs >= 2:
    img_array.append(opt.input_image2)
if number_of_inputs >= 3:
    img_array.append(opt.input_image3)
img = [Image.open(i) for i in img_array]

print("Number of inputs", number_of_inputs, " | Image size", img[0].size)

img_out = Image.open(opt.target_image)
img = [i.convert('YCbCr') for i in img]   # Converting to YCbCr
img_out = img_out.convert('YCbCr')
y = [i.split()[0] for i in img]
y = [np.lib.pad(i,pad_width=(img_pad,img_pad),mode='reflect') for i in y]   # Padding the image

y_out, cb_out, cr_out = img_out.split()
y_out = np.lib.pad(y_out, pad_width = (img_pad, img_pad), mode='reflect')          # Padding the image
w, h = img[0].size


# Retriving the Model
model = models.get_net(input_depth=1, NET_TYPE="NRRN")
model.load_state_dict(torch.load(opt.model, map_location="cuda"))


if opt.cuda:
    model = torch.nn.DataParallel(model)
model.to(device)

criterion = nn.MSELoss().to(device)

# transorming the input tensors and the target
img_to_tensor = Compose([ToTensor(),Normalize((0.0,),(1.0,))])


big_img = [img_to_tensor(i).to(device) for i in y]
big_img_target = img_to_tensor(y_out).to(device)
w, h = big_img[0].shape[1], big_img[0].shape[2]

# Patching the big images and the target image
ptchs = [patches.extract_patches_2d(i.unsqueeze(0), (patch_size_w,patch_size_w)
        ,step=[patch_size_h,patch_size_h], batch_first=True) for i in big_img]
ptchs = [ptchs[i].squeeze(0) for i in range(len(ptchs))]
target = patches.extract_patches_2d(big_img_target.unsqueeze(0), (patch_size_w,patch_size_w)
        , step=[patch_size_h,patch_size_h],batch_first=True)
target = target.squeeze(0)

#Forming batch with the images
chunks = 350//1
btch = [torch.chunk(i,chunks,0) for i in ptchs]
btch_target = torch.chunk(target,chunks,0)
print("Batch count", len(btch[0]),"| Batch shape ",btch[0][0].shape)
psnr_l = []
batch_size = btch[0][0].shape[0]

out = torch.empty(ptchs[0].shape,requires_grad=False)  #defining the output tensor

for j in range(len(btch[0])):
    index = j*batch_size

    if number_of_inputs == 1:
        x = torch.cat((btch[0][j].to(device),btch[0][j].to(device)),dim=1)
        o = model(x)
        o2 = o
    elif number_of_inputs==2:
        x = torch.cat((((btch[0][j]+btch[1][j])*0.5).to(device),btch[1][j].to(device)),dim=1)
        y = torch.cat((((btch[0][j]+btch[1][j])*0.5).to(device),btch[0][j].to(device)),dim=1)
        o = model(x)
        o2 = model(y)
    elif number_of_inputs==3:
        x = torch.cat((btch[0][j].to(device),btch[1][j].to(device)),dim=1)
        y = torch.cat((btch[0][j].to(device),btch[2][j].to(device)),dim=1)
        o = model(x)
        o2 = model(y)
    out[index:index+btch[0][j].shape[0],:,:,:] = ((o2 + o)*0.5).detach().cpu()
# Calculating the PSNR of each patch
    mse = criterion((o2 + o)*0.5, btch_target[j])
    psnr_l.append(10*log10(1/mse.item()))
    batch_size = btch[0][j].shape[0]


print("PSNR patches: mean {:.4f}, std {:.4f}".format(np.mean(psnr_l),np.std(psnr_l)))
#Stiching back the large images
im = patches.reconstruct_from_patches_2d(out.unsqueeze(0),img_shape=[w,h]
       ,step=[patch_size_h,patch_size_h],batch_first=True)

# The Stiching the Input Image
big_img = [big_img[i][:,img_pad:-1*img_pad,img_pad:-1*img_pad] for i in range(len(img))]
big_img = [Image_from_tensor(big_img[i].cpu(),255) for i in range(len(img))]
big_img[0].save("InputImage.png")           # reconstructed image
# The denoised Image
im = im[:,:,img_pad:-1*img_pad,img_pad:-1*img_pad]
numpy_img = Image_from_tensor(im.cpu(),255)
numpy_img.save("DenoisedImage.png")         # reconstructed image
print("Denoised imade saved: ","DenoisedImage.png")
print(" | Image size", numpy_img.size)

# Stiching yhe Target Image
big_img_target = big_img_target[:,img_pad:-1*img_pad,img_pad:-1*img_pad]
numpy_target = Image_from_tensor(big_img_target.cpu(),255)
numpy_target.save("TargetImage.png")        # reconstructed image

print("NRRN  PSNR: ",psnr(np.array(numpy_target),np.array(numpy_img)))
print("NRRN SSIM ",ssim(np.array(numpy_target),np.array(numpy_img), win_size=127,data_range=np.array(numpy_img).max() -
    np.array(numpy_img).min()))
print("Input Image PSNR: ",psnr(big_img[0],numpy_target))
print("Input Image SSIM ",ssim(np.array(y_out),np.array(y[0]),win_size=127,data_range=np.array(y[0]).max()-np.array(y[0]).min()))
