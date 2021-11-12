#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuente: OpenCV
@author: nacho
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from triangulation import triangulation 

#Imagenes
#imgname1 = 'imgnube1_1.jpg'
#imgname2 = 'imgnube1_2.jpg'
imgname1 = 'imgnube2_1.jpg'
imgname2 = 'imgnube2_2.jpg'

#Creamos objeto sift
sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Procesamiento de imagen en escala de grises
kp1, des1 = sift.detectAndCompute(img1,None)   #des es el descriptor

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)# Procesamiento de imagen en escala de grises
kp2, des2 = sift.detectAndCompute(img2,None)  #des es el descriptor

hmerge = np.hstack((gray1, gray2)) #Costura horizontal
cv2.imshow("gray", hmerge) #Mosaico se muestra en gris
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) # Dibuje los puntos característicos y muéstrelos como círculos rojos
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) # Dibuje los puntos característicos y muéstrelos como círculos rojos
hmerge = np.hstack((img3, img4)) #Costura horizontal
cv2.imshow("point", hmerge) #Mosaico se muestra en gris
cv2.waitKey(0)

# BFMatcher resuelve
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Ajustar relación
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None,flags=2)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

for match in good:
      x1 = kp1[match[0].queryIdx].pt
      x2 = kp2[match[0].trainIdx].pt

#Con el punto x1 y x2 haces la triangulación 
#Orientacion y posicion se obtienen de la cámara en cada momento de la foto
orientation_1 = [0 ,0 ,0]
orientation_2 = [0, 0, 0]

pos_1 = [0, 0, 0]
pos_2 = [0, 0, 0]

p_3d = triangulation(x1, x2, orientation_1, pos_1, orientation_2, pos_2)

