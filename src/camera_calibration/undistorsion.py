# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:28:37 2021

@author: EQUIPO
"""

import json
import cv2
import numpy as np

def fisheye_undistorsion(img):
    
    f = open('src/camera_calibration/fisheye_calibration_data.json')
    data = json.load(f)
    #Matriz calibración
    new_K = np.array(data['new_K'])
    K = np.array(data['K'])
    D = np.array(data['D'])
    dim3 = np.array(data['dim3'])
    scaled_K = np.array(data['scaled_K'])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

