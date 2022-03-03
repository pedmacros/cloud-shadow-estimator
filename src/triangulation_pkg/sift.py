#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuente: OpenCV
@author: nacho
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from src.camera_calibration.fisheye_triangulation import fisheye_triangulation 
from src.triangulation_pkg.contrast import contrast
from src.camera_calibration.undistorsion import  fisheye_undistorsion

#Imagenes
#imgname1 = 'img/sift_images/imgnube1_1.jpg'
#imgname2 = 'img/sift_images/imgnube1_2.jpg'
#imgname1 = 'img/sift_images/imgnube2_1.jpg'
#imgname2 = 'img/sift_images/imgnube2_2.jpg'

imgname1 = 'img/nubes_azotea/2esquina/100_0134.jpg'
imgname2 = 'img/nubes_azotea/3esquina/100_0138.jpg'

#Creamos objeto sift
sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

img_aux = cv2.imread(imgname1)
img_aux = fisheye_undistorsion(img_aux)


hmerge = np.hstack((img1, img2)) #Costura horizontal
cv2.namedWindow("Imagenes originales",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Imagenes originales", 1200,700) 
cv2.imshow("Imagenes originales", hmerge) #Mosaico se muestra en gris

cv2.waitKey(0)

cv2.imwrite('img/resultado/originales.jpg', hmerge)

img1 = fisheye_undistorsion(img1)
img2 = fisheye_undistorsion(img2)

hmerge = np.hstack((img1, img2)) #Costura horizontal
cv2.namedWindow("Imagenes sin distorsion",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Imagenes sin distorsion", 1200,700) 
cv2.imshow("Imagenes sin distorsion", hmerge) #Mosaico se muestra en gris

cv2.waitKey(0)

cv2.imwrite('img/resultado/distorsion.jpg', hmerge)

img1 = contrast(img1)
img2 = contrast(img2)

hmerge = np.hstack((img1, img2)) #Costura horizontal
cv2.namedWindow("Imagenes con contraste",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Imagenes con contraste", 1200,700) 
cv2.imshow("Imagenes con contraste", hmerge) #Mosaico se muestra en gris

cv2.waitKey(0)

cv2.imwrite('img/resultado/contraste.jpg', hmerge)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # Procesamiento de imagen en escala de grises
kp1, des1 = sift.detectAndCompute(gray1,None)   #des es el descriptor

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)# Procesamiento de imagen en escala de grises
kp2, des2 = sift.detectAndCompute(gray2,None)  #des es el descriptor


img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) # Dibuje los puntos característicos y muéstrelos como círculos rojos
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) # Dibuje los puntos característicos y muéstrelos como círculos rojos
hmerge = np.hstack((img3, img4)) #Costura horizontal
cv2.namedWindow("Imagenes SIFT",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Imagenes SIFT", 1200,700) 
cv2.imshow("Imagenes SIFT", hmerge) #Mosaico se muestra en gris

cv2.waitKey(0)

cv2.imwrite('img/resultado/sift.jpg', hmerge)

# BFMatcher resuelve
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Ajustar relación
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:],None, flags = 2)

cv2.namedWindow("Matching",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Matching", 1200,700)
cv2.imshow("Matching", img5)
cv2.waitKey(0)

cv2.imwrite('img/resultado/matching.jpg', img5)

# Initialize lists
list_kp1 = []
list_kp2 = []

orientation_1 = [0 ,0 ,0]
orientation_2 = [0, 0, 0]

pos_1 = [10.15, 0, 0]
pos_2 = [0, 4.67, 0] #10.15, 4.67

points_3d = []
points_3d_x = []
points_3d_y = []
points_3d_z = []
points_2d = []

# For each match...
for mat in good:

    # Get the matching keypoints for each of the images
    img1_idx = mat[0].queryIdx
    img2_idx = mat[0].trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    p1 = kp1[img1_idx].pt
    p2 = kp2[img2_idx].pt
    
    p_3d = fisheye_triangulation(p1, p2, orientation_1, pos_1, orientation_2, pos_2)
    if(p_3d[2] > 10 and p_3d[2]<10000):
        points_3d_z.append(p_3d[2])
        points_3d_y.append(p_3d[1])
        points_3d_x.append(p_3d[0])
        points_3d.append(p_3d)
        points_2d.append([p1])


print(np.mean(points_3d_z))

for i in range(len(points_2d)):
    img_aux = cv2.circle(img_aux, (int(points_2d[i][0][0]),int(points_2d[i][0][1])) , radius = 5, color=(255,0,255), thickness = 2)
    img_aux = cv2.putText(img_aux, str(int(points_3d[i][2])), (int(points_2d[i][0][0])-35,int(points_2d[i][0][1])+35), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0,0,255), thickness = 2)

cv2.namedWindow("Altura puntos",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Altura puntos", 1200,700)
cv2.imshow("Altura puntos", img_aux)
cv2.waitKey(0)

cv2.imwrite('img/resultado/altura.jpg', img_aux)

cv2.destroyAllWindows()