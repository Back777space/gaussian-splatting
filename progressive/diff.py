import cv2


im1 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\train\\test\\contrib_fixed\\progressive_30000_1.0\\00000.png') / 255.0
im2 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\train\\test\\voxels_fixed\\progressive_30000_1.0\\00000.png') / 255.0

diff = abs(im1 - im2) * 10

cv2.imshow('vensterke', diff)
cv2.waitKey()