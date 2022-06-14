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
from wavelet import iter_wavelet

def calculate_MSE(pre, gt):
    diff = pre - gt
    return np.mean(diff ** 2)

def calculate_MAE(pre, gt):
    diff = pre - gt
    return np.mean(abs(diff))
    

if __name__ == "__main__":
    dataset = "venus"
    #image_paths = glob.glob('./ohta/*.ppm')
    image_paths = glob.glob('./{}/*.ppm'.format(dataset))
    #image_paths = glob.glob('./test/*.JPG')
    #image_paths = glob.glob('./test2/*.jpg')
    # 2, 6
    img1 = cv2.imread(image_paths[6])
    img2 = cv2.imread(image_paths[2])
    H, W = img1.shape[0], img2.shape[1]
    plt.imsave("./left.jpg", img1[..., ::-1])
    plt.imsave("./right.jpg", img2[..., ::-1])
    
    image_paths = glob.glob('./{}/*.pgm'.format(dataset))
    gt = cv2.imread(image_paths[0])[..., 0]
    # test
# =============================================================================
#     block_size = 35
#     plt.figure("gt")
#     plt.imshow(gt / 8, cmap = "gray")
#     plt.axis("off")
#     gt = gt[block_size // 2 : -block_size // 2, block_size // 2 : - block_size // 2]
#     gt = gt / 8
#     print(gt[143, 22])
# =============================================================================
    
# =============================================================================
#     ori = np.load("./ori_venus_result_35.npy")
#     plt.figure("ori")
#     plt.imshow(ori)
#     plt.axis("off")
#     sw = np.load("./sw_venus_result_35.npy")
#     plt.figure("sw")
#     plt.imshow(sw)
#     plt.axis("off")
#         
#     print(calculate_MSE(ori, gt))
#     print(calculate_MSE(sw, gt))
# =============================================================================
    
    # solve F, T, R
    match_keypoint1, match_keypoint2 = find_match(img1, img2, topK = 150, method = "ORB")
    F = RANSAC_F(match_keypoint1, match_keypoint2, threshold = 0.0001)
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
    img1_rect, img2_rect, point1_rect, point2_rect = rectified(img1, img2, match_keypoint1, match_keypoint2, F)
# =============================================================================
#     plt.figure("img1 - rectified_CV")
#     plt.imshow(img1_rect_CV[..., ::-1])
#     plt.figure("img2 - rectified_CV")
#     plt.imshow(img2_rect_CV[..., ::-1])
# =============================================================================

    ## method 2 
# =============================================================================
#     img1_rect, point1_rect, R_rect = rectified2(img1, T, match_keypoint1)
#     gt_rect, R_rect = rectified2(gt, T)
#     img2_rect, point2_rect, R_rect = rectified2(img2, T, match_keypoint2)
# =============================================================================
# =============================================================================
#     plt.figure("img1 - rectified")
#     plt.imshow(img1_rect[..., ::-1])
#     plt.figure("img2 - rectified")
#     plt.imshow(img2_rect[..., ::-1])
#     plt.figure("gt")
#     plt.imshow(gt_rect)
# =============================================================================
    #rect_test(img1, img2, match_keypoint1, match_keypoint2, F, T)
    
    # correspond
    block_size = 35
    gt = gt / 8
    gt = gt[block_size // 2 : -block_size // 2, block_size // 2 : -block_size // 2]
    disp_map = correspond(img1_rect, img2_rect, block_size)
    disp_map = disp_map[block_size // 2 : -block_size // 2, block_size // 2 : -block_size // 2]
    print("original MSE : {:.6f}".format(calculate_MSE(disp_map, gt)))
    print("original MAE : {:.6f}".format(calculate_MAE(disp_map, gt)))
    np.save("./ori_{}_result_{}.npy".format(dataset, block_size), disp_map)
    print(img1.shape)
    SW_disp_map = SW_correspond(img1, img2, block_size, gt)
    SW_disp_map = SW_disp_map[block_size // 2 : -block_size // 2, block_size // 2 : -block_size // 2]
    print("SW MSE: {:.6f}".format(calculate_MSE(SW_disp_map, gt)))
    print("SW MAE : {:.6f}".format(calculate_MAE(SW_disp_map, gt)))
    #np.save("./sw_{}_result_{}.npy".format(dataset, block_size), SW_disp_map)
    plt.figure()
    plt.imshow(SW_disp_map)
    
    refine_disp = iter_wavelet(disp_map)
    print("original refine MSE: {:.6f}".format(calculate_MSE(refine_disp, gt)))
    print("original refine MAE : {:.6f}".format(calculate_MAE(refine_disp, gt)))
    np.save("./ori_{}_result_refine_{}.npy".format(dataset, block_size), refine_disp)
    
    SW_refine_disp = iter_wavelet(SW_disp_map)
    print("SW refine MSE: {:.6f}".format(calculate_MSE(SW_refine_disp, gt)))
    print("SW refine MAE : {:.6f}".format(calculate_MAE(SW_refine_disp, gt)))
    np.save("./sw_{}_result_refine_{}.npy".format(dataset, block_size), SW_refine_disp)
    
    
