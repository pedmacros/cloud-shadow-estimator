import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Blue level threshold to mask the sky
low_blue = np.array([100,0,0])
high_blue = np.array([255,255,255])

img_src = cv.imread("img/cumulus/img04.jpg")
img_bw = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
mask = cv.inRange(img_src, low_blue, high_blue)
res = cv.bitwise_and(img_src, img_src, mask=mask)

cv.imshow("Original", img_src)
cv.imshow("res", res)

k = cv.waitKey(0)


