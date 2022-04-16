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


def dataLoader(path):
    print("loading images from ", path)
    images_data = []
    for n in range(0, 2):
        imgs = path + "/" + "im" + str(n) + ".png"
        image = cv2.imread(imgs)
        
        if image is not None:
            images_data.append(image)			
        else:
            print("Unable to load images ", image)

    return images_data