#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#Funcion altitud
def calc_alfa_s(w, Lambda, delta):
    alfa_s = np.arcsin(np.cos(Lambda)*np.cos(w)*np.cos(delta) + np.sin(Lambda)*np.sin(delta))
    return alfa_s

#Funcion acimut
def calc_gamma_c(w, Lambda, delta, alfa_s):
    gamma_c = np.arccos((np.cos(Lambda)*np.sin(delta) - np.cos(delta)*np.sin(Lambda)*np.cos(delta))/np.cos(alfa_s))
    return gamma_c

#Angulo horario orto u ocaso
def calc_hr(Lambda, delta):
    hr = np.arccos(-np.tan(Lambda)*np.tan(delta))
    return hr

def posicion_solar(dia, hora):

    #Parametros necesarios
    #Lambda = 35.79994 * np.pi/180 #Latitud
    Lambda = 37.09464396122538 * np.pi/180 #Latitud
    #Lambda = 37.45769 * np.pi/180
    h = 12
    #OJO VER FORMATO TIME PYTHON
    Dia = dia

    #Declinacion
    x = 2*np.pi*(Dia-1+(h-12)/24)/365
    delta = 0.006918 - 0.399912*np.cos(x) + 0.070257*np.sin(x) - 0.006758*np.cos(2*x) 
    + 0.000907*np.sin(2*x) - 0.002697*np.cos(3*x) + 0.001480*np.sin(3*x)
    
    #Hora ocaso
    hMax = calc_hr(Lambda, delta)*12/np.pi
    
    #Hora en radianes
    Hora = hora - 12
    w = Hora * np.pi/12
    
    #Altitud
    alfa_s = calc_alfa_s(w, Lambda, delta)
    
    #Acimut
    if Hora > 0:
         gamma_c = 180 - (2*np.pi-calc_gamma_c(w,Lambda,delta,alfa_s))*180/np.pi
    else:
         gamma_c = 180 - (calc_gamma_c(w,Lambda,delta,alfa_s)*180/np.pi)

    gamma_c = gamma_c 
    alfa_s = alfa_s * 180/np.pi
    
    print(alfa_s, gamma_c)