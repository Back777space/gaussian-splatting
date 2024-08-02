import cv2


im1 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\truck\\test\\voxels_contr_antimatter_65\\progressive_30000_0.1\\00000.png') / 255.0
im2 = cv2.imread('C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models\\truck\\test\\frustum_voxels_65\\progressive_30000_0.1\\00000.png') / 255.0

diff = abs(im1 - im2) * 10

cv2.imshow('vensterke', diff)
cv2.waitKey()