#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuente: OpenCV
@author: nacho
"""
import cv2
import numpy as np

def triangulation(x1, x2, orientation_1, pos_1, orientation_2, pos_2):

    #Matriz calibración
    mtx=np.array([[1.54272784e+03, 0, 9.57409444e+02],
    [0, 1.54656053e+03, 6.29269967e+02],
    [0, 0, 1]])
    dist=np.array([0.01737745, -0.36542463, 0.01312702, -0.00285169,
    0.80215804])

    #Matriz mejorada
    h,  w = 1080, 1920
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    intrinsic=np.array(newcameramtx)

    ############################################################################################

    #Parametros imagen 1
    roll_1 = orientation_1[0]
    pitch_1 = orientation_1[1]
    yaw_1 = orientation_1[2]
    x_1 = pos_1[0]
    y_1 = pos_1[1]
    z_1 = pos_1[2]

    #Matriz de rotación imagen 1
    rotx_1=np.array([[1, 0, 0], [0, np.cos(roll_1), -np.sin(roll_1)], [0, np.sin(roll_1), np.cos(roll_1)]])
    roty_1=np.array([[np.cos(pitch_1), 0, np.sin(pitch_1)], [0, 1, 0], [-np.sin(pitch_1), 0, np.cos(pitch_1)]])
    rotz_1=np.array([[np.cos(yaw_1), -np.sin(yaw_1), 0], [np.sin(yaw_1), np.cos(yaw_1), 0], [0, 0, 1]])
    rot_1=np.matmul(rotz_1,roty_1)
    rot_1=np.matmul(rot_1,rotx_1)

    #Matriz de translación imagen 1
    trans_1=np.array([[x_1, y_1, z_1]])

    #Matriz extrínseca imagen 1
    extrinsic_1=np.concatenate((rot_1, trans_1.T), axis=1)

    #Matriz resultante imagen 1
    mat_1=np.matmul(intrinsic,extrinsic_1)

    #########################################################################################

    #Parametros imagen 2
    roll_2 = orientation_2[0]
    pitch_2 = orientation_2[1]
    yaw_2 = orientation_2[2]
    x_2 = pos_2[0]
    y_2 = pos_2[1]
    z_2 = pos_2[2]

    #Matriz de rotación imagen 2
    rotx_2=np.array([[1, 0, 0], [0, np.cos(roll_2), -np.sin(roll_2)], [0, np.sin(roll_2), np.cos(roll_2)]])
    roty_2=np.array([[np.cos(pitch_2), 0, np.sin(pitch_2)], [0, 1, 0], [-np.sin(pitch_2), 0, np.cos(pitch_2)]])
    rotz_2=np.array([[np.cos(yaw_2), -np.sin(yaw_2), 0], [np.sin(yaw_2), np.cos(yaw_2), 0], [0, 0, 1]])
    rot_2=np.matmul(rotz_2,roty_2)
    rot_2=np.matmul(rot_2,rotx_2)

    #Matriz de translación imagen 2
    trans_2=np.array([[x_2, y_2, z_2]])

    #Matriz extrínseca imagen 2
    extrinsic_2=np.concatenate((rot_2, trans_2.T), axis=1)

    #Matriz resultante imagen 2
    mat_2=np.matmul(intrinsic,extrinsic_2)

    #Triangulación
    p_3d = cv2.triangulatePoints(mat_1, mat_2, x1, x2)
    p_3d/=p_3d[3]

    print('Punto 3D calculado:',  p_3d.T)

    return p_3d