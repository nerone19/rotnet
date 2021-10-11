# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:49:35 2021

@author: gabri
"""


import cv2 
from skimage.transform import rotate,resize
import matplotlib.pyplot as plt

image =  cv2.imread('D:/additional datasets/OCR/final dataset/real screen/original dataset/Image-3552.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.resize(image,(512,512))

for degree in range(0,90,2):
    rotated_image = rotate(image, -degree, resize=True)
    plt.imshow(rotated_image)
    plt.show()