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

def estimateCameraPose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    rotation = []
    translation = []

    rotation.append(np.dot(U, np.dot(W, V_T)))
    rotation.append(np.dot(U, np.dot(W, V_T)))
    rotation.append(np.dot(U, np.dot(W.T, V_T)))
    rotation.append(np.dot(U, np.dot(W.T, V_T)))
    translation.append(U[:, 2])
    translation.append(-U[:, 2])
    translation.append(U[:, 2])
    translation.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(rotation[i]) < 0):
            rotation[i] = -rotation[i]
            translation[i] = -translation[i]

    return rotation, translation

def getPositiveCount(pts3D, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    pos = 0
    
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        Xc = np.dot(P, X)
        Xc = Xc / Xc[3]
        z = Xc[2]
        if z > 0:
            pos += 1

    return pos

def calcBestRT(pts_3D, rotation, translation):
    c_1 = []
    c_2 = []

    rot = np.identity(3)
    tran = np.zeros((3,1))
    for i in range(len(pts_3D)):
        pts3D = pts_3D[i]
        pts3D = pts3D/pts3D[3, :] 
        c_2.append(getPositiveCount(pts3D, rotation[i], translation[i]))
        c_1.append(getPositiveCount(pts3D, rot, tran))

    c_1 = np.array(c_1)
    c_2 = np.array(c_2)

    c_t = int(pts_3D[0].shape[1] / 2)
    idx = np.intersect1d(np.where(c_1 > c_t), np.where(c_2 > c_t))
    r_best = rotation[idx[0]]
    t_best = translation[idx[0]]

    return r_best, t_best

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, cc = img1.shape

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    
    return img1,img2