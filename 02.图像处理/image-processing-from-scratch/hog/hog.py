import numpy as np
import cv2

img = cv2.imread('lena.png')
img = np.float32(img) / 255.0
cv2.imshow('origin', img)
cv2.waitKey()

# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

cv2.imshow('gx', gx)
cv2.waitKey()

cv2.imshow('gy', gy)
cv2.waitKey()

mag, angle = cv2.cartToPolar(gx, gy, 1)
cv2.imshow('mag', mag)
cv2.waitKey()

cv2.imshow('angle', angle)
cv2.waitKey()
