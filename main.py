# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:55:38 2022

@author: Eddie
"""

from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
from func import find_match, RANSAC_F, solve_translation, solve_rotation, normalize_point, solve_fundamental_matrix
from rectified import rectified, rectified2, rectified22, rect_test
from correspondence import correspond, SW_correspond


if __name__ == "__main__":
    image_paths = glob.glob('./barn1/*.ppm')
    #image_paths = glob.glob('./test3/*.JPG')
    #image_paths = glob.glob('./test2/*.jpg')
    img1 = cv2.imread(image_paths[0])
    img2 = cv2.imread(image_paths[1])
    plt.imsave("./left.jpg", img1[..., ::-1])
    plt.imsave("./right.jpg", img2[..., ::-1])
    
    # solve F, T, R
    match_keypoint1, match_keypoint2 = find_match(img1, img2, topK = 150, method = "ORB")
    #match_keypoint1, match_keypoint2, T1, T2 = normalize_point(img1, img2, match_keypoint1, match_keypoint2)
    #F = solve_fundamental_matrix(match_keypoint1, match_keypoint2)
    F = RANSAC_F(match_keypoint1, match_keypoint2, threshold = 1e-3)
    T = solve_translation(F, match_keypoint1, match_keypoint2)
    R = solve_rotation(F, T)
    print("F : \n", F)
    print("T : \n", T)
    print("R : \n", R)
    print("det(R) : ", np.linalg.det(R))
    # rectified the image
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ## method 1 (opencv)
    #img1_rect, img2_rect, point1_rect, point2_rect = rectified(img1, img2, match_keypoint1, match_keypoint2, F)
    ## method 2 
    img1_rect, point1_rect, R_rect = rectified2(img1, match_keypoint1, T)
    img2_rect, point2_rect, R_rect = rectified2(img2, match_keypoint2, T)
# =============================================================================
#     plt.figure("img1 - rectified")
#     plt.imshow(img1_rect[..., ::-1])
#     plt.figure("img2 - rectified")
#     plt.imshow(img2_rect[..., ::-1])
# =============================================================================
    #rect_test(img1, img2, match_keypoint1, match_keypoint2, F, T)
    
    # correspond
    SW_correspond(img1_rect, img2_rect)
    image_paths = glob.glob('./barn1/*.pgm')
    gt1 = cv2.imread(image_paths[0])
# =============================================================================
#     plt.figure()
#     plt.imshow(gt1)
# =============================================================================
    
    
# =============================================================================
#     plt.figure()
#     plt.imshow(img2_rect[..., ::-1])
#     plt.figure()
#     plt.imshow(img1[..., ::-1])
#     _, H1, H2 = cv2.stereoRectifyUncalibrated(match_keypoint1, match_keypoint2, F, imgSize=img1.shape[:2])
#     print(H1)
#     print(H2)
# =============================================================================
    
    
    
