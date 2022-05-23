# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:12:16 2022

@author: Eddie
"""

import cv2
import numpy as np

def rectified(img1, img2, point1, point2, F):
    h1, w1 = img1.shape[0], img1.shape[1]
    h2, w2 = img2.shape[0], img2.shape[1]
    
    point1_rect = np.zeros((point1.shape))
    point2_rect = np.zeros((point2.shape))    
    _, H1, H2 = cv2.stereoRectifyUncalibrated(point1, point2, F, imgSize=(w1, h1))
    for i in range(point1.shape[0]):
        tmp = np.dot(H1, point1[i])
        point1_rect[i] = tmp / tmp[-1]
        tmp = np.dot(H2, point2[i])
        point2_rect[i] = tmp / tmp[-1]
    
    img1_rect = cv2.warpPerspective(img1, H2, (w1, h1))
    img2_rect = cv2.warpPerspective(img2, H1, (w2, h2))        
    
    return img1_rect, img2_rect, point1_rect, point2_rect

def rectified2(img, point, T):
    e1 = T / np.sum(T ** 2)
    e2 = np.array([-T[2], T[0], 0]) / np.sqrt(T[0] ** 2 + T[1] ** 2)
    e3 = np.cross(e1, e2)
    R_rect = np.array([e1, e2, e3])
    
    img_rect = cv2.warpPerspective(img, R_rect, (img.shape[1], img.shape[0]))
    point = np.dot(point, R_rect)
    point = point / point[:, -1:]
    return img_rect, point, R_rect
