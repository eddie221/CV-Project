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
from func import find_match, RANSAC_F, solve_translation, solve_rotation, draw_epipole_lines
from rectified import rectified, rectified2


if __name__ == "__main__":
    image_paths = glob.glob('./barn1/*.ppm')
    #image_paths = glob.glob('./test/*.JPG')
    #image_paths = glob.glob('./test2/*.jpg')
    img1 = cv2.imread(image_paths[0])
    img2 = cv2.imread(image_paths[1])
# =============================================================================
#     img1 = cv2.resize(img1, (512, 256))
#     img2 = cv2.resize(img2, (512, 256))
# =============================================================================
    match_keypoint1, match_keypoint2 = find_match(img1, img2)
    F = RANSAC_F(match_keypoint1, match_keypoint2)
    T = solve_translation(F, match_keypoint1, match_keypoint2)
    R = solve_rotation(F, T)
    print("T : \n", T)
    print("R : \n", R)
    img1_rect, img2_rect, point1_rect, point2_rect = rectified(img1, img2, match_keypoint1, match_keypoint2, F)
    #R_rect = rectified2(img2, match_keypoint2, T)
    
    
    # check retified result
    lines1 = cv2.computeCorrespondEpilines(point2_rect[:, :2].reshape(-1, 1, 2), 2, F)
    lines2 = cv2.computeCorrespondEpilines(point1_rect[:, :2].reshape(-1, 1, 2), 1, F)
# =============================================================================
#     lines1 = np.matmul(point2_rect, F)
#     lines2 = np.matmul(F, point1_rect.T).T
# =============================================================================
    lines1 = lines1.reshape(-1,3)
    lines2 = lines2.reshape(-1,3)
    img1 = draw_epipole_lines(img1, lines1, point1_rect[:, :2])
    img2 = draw_epipole_lines(img2, lines2, point2_rect[:, :2])
    plt.figure()
    plt.imshow(img1[..., ::-1])
    plt.figure()
    plt.imshow(img2[..., ::-1])
    
    
# =============================================================================
#     plt.figure()
#     plt.imshow(img2_rect[..., ::-1])
#     plt.figure()
#     plt.imshow(img1[..., ::-1])
#     _, H1, H2 = cv2.stereoRectifyUncalibrated(match_keypoint1, match_keypoint2, F, imgSize=img1.shape[:2])
#     print(H1)
#     print(H2)
# =============================================================================
    
    
    
