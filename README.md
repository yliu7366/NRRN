# Denoising

## OHSU Dataset

The goal of this project is to denoise images acquired with FIB-SEM.
The images are of size `6144x4096` pixels, 8-bit, 24 MB.
The currently available data set (OHSU dataset) consists of 5 stacks (volumes), sliced into 10 images (Input Images) and the same number of noise-free
"clean" images (labels), total 100 images.
For the training and validating phase we have patched the original images into 256x256 patches. The patches from stacks 1 to 3 were used for training, stack 4 for testing and stack 5 for inference.
That resulted in a training set of 8190 images and  testing 2730.
The dataset is expected to be in folder Data/


## EPFL Dataset

The [EPFL dataset](https://www.epfl.ch/labs/cvlab/data/data-em/) corresponds to a `1065x2048x1536` volume.
Each sub-volume consists of the first 165 slices of the `1065x2048x1536` image stack.
The volume used for training is the top part of the image stack while the bottom part is to be used for testing.
In the case of EPFL the dataset consist of only one image realization,
we add additional Gaussian and/or Poisson noise.

## Organizing the Dataset

###OHSU Dataset

The dataset needs to be organized as
```sh
                    ___Inputs_1_1_1_1.png
        ___inputs--|___Inputs_1_1_1_2.png
       |           |___ ......
train--|            ___Outputs_1_1_1_1.png
       |___labels--|___Outputs_1_1_1_2.png
                   |___......

      ___inputs
     |
val--|
     |___labels
```
and the images named as Inputs_#1_#2_#3_#4.png
where #1 is the image stack  (volume) number, #2 is the slice number in the
stack, #3 and #4 are the patch numbers. The images in the labels folder are only used
in calculating PSNR for validation only (not used in the denoising process).

### EPFL Dataset

The dataset needs to be organized as
```sh
        ___image_0.png
       |___image_1.png
train--|___image_2.png
       |___image_165.png

        ___image_0.png
       |___image_1.png
val  --|___image_2.png
       |___image_165.png

```

## Training Command

To test if the script is working on randomly generated data samples
```sh
python3 main.py --cuda --dataset TEST
```
or to train on OHSU/EPFL dataset
```sh
python3 main.py --cuda --nEpoch 30 --data_dir "path/to/the/dataset/" --dataset <OHSU/EPFL>
```
The model is going to be saved in the working directory under the name
`Bestmodel_epoch_#Epoch_Number.pth`.


## Inference Command

`denoise_infer.py` script will take in high resolution input image triplets, extract overlapping
crops from them and then denoise the extracted triplets. After obtaining the denoised output,
it will stitch back all the crops into one final denoised image (the noisy input image, the
denoised image and the ground truth image will be saved in your current directory).
It will also display the PSNR and SSIM value.

Run inference using input triplets. Specify them using `input_image(i)` arguements.
```sh
python3 denoise_infer.py --cuda --input_image SliceImage005-ICD-09.png
--input_image2 SliceImage005-ICD-08.png --input_image3 SliceImage005-ICD-10.png --target_image
SliceImage005-09.png --model Bestmodel_epoch_87.pth --overlap 20
```

The denoised image is saved in the working directory under the name
`DenoisedImage.png`.
