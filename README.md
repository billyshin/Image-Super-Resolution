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
    {\begin{aligned}{\mathit  {PSNR}}&=10\cdot \log _{{10}}\left({\frac  {{\mathit  {MAX}}_{I}^{2}}{{\mathit  {MSE}}}}\right)\\&=20\cdot \log _{{10}}\left({\frac  {{\mathit  {MAX}}_{I}}{{\sqrt  {{\mathit  {MSE}}}}}}\right)\\&=20\cdot \log _{{10}}\left({{\mathit  {MAX}}_{I}}\right)-10\cdot \log _{{10}}\left({{{\mathit  {MSE}}}}\right)\end{aligned}}
