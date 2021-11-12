# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:39:25 2021

@author: EQUIPO
"""

import cv2

def contrast(img1):
    # #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)
    # cv2.waitKey(0)
    
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    # cv2.waitKey(0)
    
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)
    # cv2.waitKey(0)
    
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    # cv2.imshow('limg', limg)
    # cv2.waitKey(0)
    
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)
    return(final)
