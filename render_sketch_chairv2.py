import numpy as np
from bresenham import bresenham
import scipy.ndimage
import random


def mydrawPNG(vector_images, Sample = 25, Side = 256):
    for vector_image in vector_images:
        pixel_length = 0
        #number_of_samples = random.
        sample_freq = list(np.round(np.linspace(0,  len(vector_image), 18)[1:]))
        Sample_len = []
        raster_images = []
        raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
        initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
        for i in range(0, len(vector_image)):
            if i > 0: 
                if vector_image[i-1, 2] == 1:
                    initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

            cordList = list(bresenham(initX, initY, int(vector_image[i,0]), int(vector_image[i,1])))
            pixel_length += len(cordList)

            for cord in cordList:
                if (cord[0] > 0 and  cord[1] > 0) and (cord[0] < Side and  cord[1] < Side):
                    raster_image[cord[1], cord[0]] = 255.0
            initX , initY = int(vector_image[i,0]), int(vector_image[i,1])

            if i in sample_freq:
                raster_images.append(scipy.ndimage.binary_dilation(raster_image) * 255.0)
                Sample_len.append(pixel_length)

        raster_images.append(scipy.ndimage.binary_dilation(raster_image) * 255.0)
        Sample_len.append(pixel_length)

    return raster_images, Sample_len


def Preprocess_QuickDraw_redraw(vector_images, side = 256.0):
    vector_images = vector_images.astype(np.float)
    vector_images[:, :2] = vector_images[:, :2] / np.array([256, 256])
    vector_images[:,:2] = vector_images[:,:2] * side
    vector_images = np.round(vector_images)
    return vector_images

def redraw_Quick2RGB(vector_images):
    vector_images_C = Preprocess_QuickDraw_redraw(vector_images)
    raster_images, Sample_len = mydrawPNG([vector_images_C])
    return raster_images,  Sample_len
