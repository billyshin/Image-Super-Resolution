# Image-Super-Resolution

## Using the Super Resolution Convolutional Neural Network for Image Restoration
   - This network was published in the paper, "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong in 2014.
   - We will be deploying the super-resolution convolution nerual network (SRCNN) using Keras.
   - We can use this model to improve the image quality of low resolution images. 
    
## Step 1: Necessary Packages
    - sys
    - keras
    - cv2
    - tensorflow
    - numpty
    - matplotlib
    - skimage
    - os
    
## Step 2: Image Quality Metrics
Calculate the following:
    - Peak Signal-to-Noise Ratio (PSNR)
    
   <a href="https://www.codecogs.com/eqnedit.php?latex={\begin{aligned}{\mathit&space;{PSNR}}&=10\cdot&space;\log&space;_{{10}}\left({\frac&space;{{\mathit&space;{MAX}}_{I}^{2}}{{\mathit&space;{MSE}}}}\right)\\&=20\cdot&space;\log&space;_{{10}}\left({\frac&space;{{\mathit&space;{MAX}}_{I}}{{\sqrt&space;{{\mathit&space;{MSE}}}}}}\right)\\&=20\cdot&space;\log&space;_{{10}}\left({{\mathit&space;{MAX}}_{I}}\right)-10\cdot&space;\log&space;_{{10}}\left({{{\mathit&space;{MSE}}}}\right)\end{aligned}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\begin{aligned}{\mathit&space;{PSNR}}&=10\cdot&space;\log&space;_{{10}}\left({\frac&space;{{\mathit&space;{MAX}}_{I}^{2}}{{\mathit&space;{MSE}}}}\right)\\&=20\cdot&space;\log&space;_{{10}}\left({\frac&space;{{\mathit&space;{MAX}}_{I}}{{\sqrt&space;{{\mathit&space;{MSE}}}}}}\right)\\&=20\cdot&space;\log&space;_{{10}}\left({{\mathit&space;{MAX}}_{I}}\right)-10\cdot&space;\log&space;_{{10}}\left({{{\mathit&space;{MSE}}}}\right)\end{aligned}}" title="{\begin{aligned}{\mathit {PSNR}}&=10\cdot \log _{{10}}\left({\frac {{\mathit {MAX}}_{I}^{2}}{{\mathit {MSE}}}}\right)\\&=20\cdot \log _{{10}}\left({\frac {{\mathit {MAX}}_{I}}{{\sqrt {{\mathit {MSE}}}}}}\right)\\&=20\cdot \log _{{10}}\left({{\mathit {MAX}}_{I}}\right)-10\cdot \log _{{10}}\left({{{\mathit {MSE}}}}\right)\end{aligned}}" /></a>
    
    
   - Mean Squared Error (MSE)
   
   <a href="https://www.codecogs.com/eqnedit.php?latex={\mathit&space;{MSE}}={\frac&space;{1}{m\,n}}\sum&space;_{{i=0}}^{{m-1}}\sum&space;_{{j=0}}^{{n-1}}[I(i,j)-K(i,j)]^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\mathit&space;{MSE}}={\frac&space;{1}{m\,n}}\sum&space;_{{i=0}}^{{m-1}}\sum&space;_{{j=0}}^{{n-1}}[I(i,j)-K(i,j)]^{2}" title="{\mathit {MSE}}={\frac {1}{m\,n}}\sum _{{i=0}}^{{m-1}}\sum _{{j=0}}^{{n-1}}[I(i,j)-K(i,j)]^{2}" /></a>
   
   - The structural similiarity (SSIM) index imported directly from the scikit-image library
   
## Step 3: Preparing Images
we want to produce low resolution versions of images. We can accomplish this by resizing the images, both downwards and upwards, using OpeCV. There are several interpolation methods that can be used to resize images; however, we will be using bilinear interpolation.

## Step 4: Testing Low Resolution Images
Calculate the PSNR, MSE, and SSIM between our reference images and the degraded images that we just prepared to ensure that our image quality metrics are being calculated correctly and that the images were effectively degraded.
Sample:

    bird.bmp
    PSNR: 32.8966447287
    MSE: 100.123758198
    SSIM: 0.953364486603

## Step 5: Building the SRCNN Model
Using Keras library to build the SRCNN. In Keras, it's as simple as adding layers one after the other. The achitecture and hyper parameters of the SRCNN network can be obtained from the publication referenced above.

## Step 6: Deploying the SRCNN
First of all, it will be necessary to preprocess the images extensively before using them as inputs to the network. This processing will include cropping and color space conversions. 
We will use pre-trained weights for the SRCNN in Keras for this model.
Then we can perform single-image super-resolution on our input images. Furthermore, after processing, we can calculate the PSNR, MSE, and SSIM on the images that we produce. Finally, we can display the low resolution, and high resolution images for comparison.

Sample result:

![alt text](https://github.com/billyshin/Image-Super-Resolution/blob/master/ScreenShot-2018-12-22-at-8.22.44PM.png)

