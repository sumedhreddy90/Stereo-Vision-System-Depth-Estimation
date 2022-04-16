import sys
import os
import cv2
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from scipy import fft, ifft
from numpy import histogram_bin_edges, linalg as LA
from os.path import isfile, join


def calcFundamentalMatrix(x_1, x_2):
    normalization = True
    if x_1.shape[0] > 7:
        if normalization == True:
            x1_norm, T1 = calcNormalization(x_1)
            x2_norm, T2 = calcNormalization(x_2)
        else:
            x1_norm,x2_norm = x_1,x_2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        # Calculating SVD 
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)
        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))
        if normalization:
            F = np.dot(T2.T, np.dot(F, T1))
        return F
        
    else:
        return None

def calcNormalization(uv):
    
    uv_d = np.mean(uv, axis=0)
    u_d ,v_d = uv_d[0], uv_d[1]

    u_cap = uv[:,0] - u_d
    v_cap = uv[:,1] - v_d

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_d],[0,1,-v_d],[0,0,1]])
    T = T_scale.dot(T_trans)
    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def dataLoader(path):
    images_data = []
    for n in range(0, 2):
        imgs = path + "/" + "im" + str(n) + ".png"
        image = cv2.imread(imgs)
        if image is not None:
            images_data.append(image)			
        else:
            print("Unable to load images ", image)

    return images_data