# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:13:26 2021

@author: nacho
"""

# =============================================================================
#   Esta función calcula una estimación de la posición real de un píxel que
#   no hayamos triangulado a partir de uno el cual sí conocemos su posición
#   real. Consideramos que la distancia Z es constante entre los dos puntos.
# =============================================================================
    
def pixel_distance_estimator(pixel_ref, punto_3D_ref, pixel_des):
    
    #ax y ay son los valores de distancia focal de la matriz de calibración
    ax = 1
    ay = 1
    
    #La distancia es la Z del punto referencia (coordenadas cámara)
    depth = punto_3D_ref[2]
    
    #Cálculo de la delta en coordenadas x 
    u0 = pixel_ref[0]
    u = pixel_des[0]
    delta_X_real = (depth * (u - u0))/ax
    
    #Cálculo de la delta en coordenadas y
    v0 = pixel_ref[1]
    v = pixel_des[1]
    delta_Y_real = (depth * (v - v0))/ay
    
    #Cálculo del punto real (coordenadas cámara)
    punto_3D_des = []
    punto_3D_des[0] = punto_3D_ref[0] + delta_X_real
    punto_3D_des[1] = punto_3D_ref[1] + delta_Y_real
    punto_3D_des[2] = depth
    
    return punto_3D_des
    
    