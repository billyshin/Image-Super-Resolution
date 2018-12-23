"""
Using The Super Resolution Convolutional Neural Network for Image Restoration

This network was published in the paper, "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong in 2014.

We will be deploying the super-resolution convolution nerual network (SRCNN) using Keras.

We can use this model to improve the image quality of low resolution images. 
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt 
import cv2
import numpy as np 
import math
import os

# ================================================= Image Quality Metrics =================================================
"""
A function for peak signal-to-noise ratio (PSNR)
"""
def psnr(target, reference):
    # assume the image is RGB image
    target_data = target.astype(float)
    reference_data = reference.astype(float)

    difference = reference_data - target_data
    difference = difference.flatten('C')

    rmse = math.sqrt(np.mean(difference ** 2.))

    return 20 * math.log10(255. / rmse)


"""
A function for mean square error (MSE)
"""
def mse(target, reference):
    # the MSE between the two images is the sum of the squared difference between the two images
    error = np.sum((target.astype('float') - reference.astype('float')) ** 2)
    error /= float(target.shape[0] * target.shape[1])
    return error


"""
A function that combines all three images quality metrics
"""
def compare_images(target, reference):
    scores = []
    scores.append(psnr(target, reference))
    scores.append(mse(target, reference))
    # Structural Similarity Index (SSIM) imported from scikit-image library
    scores.append(ssim(target, reference, multichannel=True))
    return scores


# ================================================= Preparing Images =================================================
"""
Prepare degraded images by introducing quality distortions via resizing.

We want to produce low resolution versions of images, and we can accomplish this by resing the images, both downwards and upwards using bilinear interpolation in OpenCV.
"""
def prepare_image(path, factor):
    # loop through the files in the directory
    for file in os.listdir(path):
        # open the file 
        image = cv2.imread(path + '/' + file)

        # find old and new image dimensions
        height, width, _ = image.shape
        new_height = height / factor
        new_width = width / factor

        # resize the image - down
        image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

        # resize the image - up
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_LINEAR)

        # save the image
        cv2.imwrite('images/{}'.format(file), image)


# ================================================= Testing Low Resolution Images =================================================
"""
Test the generated images using the image quality metrics.

To ensure that our image quality metrics are being calculated correctly and that the images were effectively degraded, lets calculate the PSNR,
MSE, and SSIM between our reference images and the degraded images that we just prepared.
"""
for file in os.listdir('images/'):
    # open target and regerence images
    target = cv2.imread('images/{}'.format(file))
    reference = cv2.imread('source/{}'.format(file))

    # calculate score
    scores = compare_images(target, reference)

    # print all three scores
    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))


# ================================================= Building the SRCNN Model =================================================
"""
Define the SRCNN Model.

SRCNN model can be built by using Keras, which adds layers one after the other.
"""
def model():
     # define model type
    SRCNN = Sequential()
    
    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    # define optimizer
    adam = Adam(lr=0.0003)
    
    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN


# ================================================= Deploying the SRCNN =================================================
"""
Define necessary image processing functions

It will be necessary to preprocess the images extensively before using them as inputs to the network. This processing will include cropping and color space conversions.
"""
def modcrop(image, scale):
    temp_size = image.shape
    size = temp_size[0:2]
    size = size - np.mod(size, scale)
    image = image[0:size[0], 1:size[1]]
    return image


def shave(image, border):
    image = image[border: -border, border: -border]
    return image
    

"""
Main prediction funcion

Used pre-trained weights for the SRCNN provided in Keras.
"""

def predict(image_path):
    # load the srcnn model with weights
    srcnn = model()
    srcnn.load_weights('3051crop_weight_200.h5')
    
    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    reference = cv2.imread('source/{}'.format(file))
    
    # preprocess the image with modcrop
    reference = modcrop(reference, 3)
    degraded = modcrop(degraded, 3)
    
    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    
    # create image slice and normalize  
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    
    # perform super-resolution with srcnn
    pre = srcnn.predict(Y, batch_size=1)
    
    # post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    
    # copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    
    # remove border from reference and degraged image
    reference = shave(reference.astype(np.uint8), 6)
    degraded = shave(degraded.astype(np.uint8), 6)
    
    # image quality calculations
    scores = []
    scores.append(compare_images(degraded, reference))
    scores.append(compare_images(output, reference))
    
    # return images and scores
    return reference, degraded, output, scores


"""
Perform single-image super-resolution on all of our input images. 

After processing, we can calculate the PSNR, MSE, and SSIM on the images that we produce.

We can save these images directly or create subplots to conveniently display the original, low resolution, and high resolution images.
"""
ref, degraded, output, scores = predict('images/flowers.bmp')

# print all scores for all images
print('Degraded Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[0][0], scores[0][1], scores[0][2]))
print('Reconstructed Image: \nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[1][0], scores[1][1], scores[1][2]))

# display images as subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
axs[1].set_title('Degraded')
axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[2].set_title('SRCNN')

# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

for file in os.listdir('images'):
    
    # perform super-resolution
    ref, degraded, output, scores = predict('images/{}'.format(file))
    
    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[1].set(xlabel = 'PSNR: {}\nMSE: {} \nSSIM: {}'.format(scores[0][0], scores[0][1], scores[0][2]))
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    axs[2].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(scores[1][0], scores[1][1], scores[1][2]))

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
      
    print('Saving {}'.format(file))
    fig.savefig('output/{}.png'.format(os.path.splitext(file)[0])) 
    plt.close()