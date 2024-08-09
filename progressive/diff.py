import cv2
import numpy as np

im1 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\truck\\test\\contrib_depth_2\\progressive_30000_0.2\\00012.png', cv2.IMREAD_GRAYSCALE) / 255.0
im2 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\truck\\test\\contrib_depth_1\\progressive_30000_0.2\\00012.png', cv2.IMREAD_GRAYSCALE) / 255.0

# im3 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\truck\\test\\frustum_contrib\\progressive_30000_1.0\\00000.png', cv2.IMREAD_GRAYSCALE) / 255.0
# im4 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\truck\\test\\frustum_contrib\\progressive_30000_0.2\\00000.png', cv2.IMREAD_GRAYSCALE) / 255.0


diff = abs(im1 - im2) 

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).mean()
    return 20 * np.log10(1.0 / np.sqrt(mse))

v = psnr(im1, im2)
# v2 = psnr_masked(im3, im4) / 255

print(v)
cv2.imshow('vensterke', diff)
# cv2.imshow('vensterke2', v2)
cv2.waitKey()