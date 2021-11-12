import matplotlib.pyplot as plt
import numpy as np
import cv2
from cloud_recognition import cloud_recognition

# Blue level threshold to mask the sky
low_blue = np.array([100,0,0])
high_blue = np.array([255,255,255])

img_src = cv2.imread("img/cirrus/img03.jpg")
img_bw = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(img_src, low_blue, high_blue)
res = cv2.bitwise_and(img_src, img_src, mask=mask)

final = cloud_recognition(img_src)

cv2.imshow("Original", img_src)
cv2.imshow("res", final)

k = cv2.waitKey(0)



