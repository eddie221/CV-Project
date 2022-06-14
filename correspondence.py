# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:41:42 2022

@author: Eddie
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2

def sum_square_error(target, source):
    return np.sum((target - source) ** 2)

def cosine_similarity(target, source):
    return np.dot(target, source) / (np.sqrt(np.sum(target ** 2)) * np.sqrt(np.sum(source ** 2)))

def similarity_strength(patch, center):
    delta_s = np.sqrt(np.sum((patch - center) ** 2, axis = -1))
    return np.exp(-delta_s / 14)

def proximity_strength(r):
    x, y = np.meshgrid(np.linspace(0, r - 1, r), np.linspace(0, r - 1, r))
    diff_x = abs(x - r // 2)
    diff_y = abs(y - r // 2)
    return np.exp(-np.sqrt((diff_x ** 2 + diff_y ** 2)) / (r / 2))


def SW_block_match(s_x, s_y, img1, img1_LAB, img2, img2_LAB, block_size, x_range, y_range, weight1, p_strength):
    threshold = 40
    x_start = max(block_size // 2, s_x - x_range // 2)
    y_start = max(block_size // 2, s_y - y_range // 2)
    x_end = min(img2_LAB.shape[1] - block_size // 2, s_x + x_range // 2 + 1)
    y_end = min(img2_LAB.shape[0] - block_size // 2, s_y + y_range // 2 + 1)
    min_center_x = None
    min_dist = np.inf
    min_source = None
    ddd = []
    fig_tmp = []
    #print(x_start, x_end, y_start, y_end)
    for j in range(y_start, y_end):
        for i in range(x_start, x_end):
            #print(j - block_size // 2, j + block_size // 2 + 1, i - block_size // 2, i + block_size // 2 + 1)
            source_LAB = img2_LAB[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            source = img2[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            weight2 = similarity_strength(source_LAB, img2_LAB[j, i]) * p_strength
            e = np.sum(abs(img1 - source), axis = -1)
            #e = np.sum((img1 * source), axis = -1) / (np.sqrt(np.sum(img1 ** 2, axis = -1)) * np.sqrt(np.sum(source ** 2, axis = -1)) + 1e-10)
            #e[e > threshold] = threshold
            distance = np.sum(e * weight1 * weight2) / (np.sum(weight1 * weight2))
            ddd.append(distance)
            fig_tmp.append(source)
# =============================================================================
#             if s_x == 22 and s_y == 143:
#                 print(distance)
#                 fig, ax = plt.subplots(3, 3)
#                 ax[0, 0].set_title("e")
#                 ax[0, 0].imshow(e)
#                 ax[0, 0].axis('off')
#                 
#                 ax[0, 1].set_title("img1")
#                 ax[0, 1].imshow(img1.astype(np.uint8))
#                 ax[0, 1].axis('off')
#                 
#                 ax[0, 2].set_title("source")
#                 ax[0, 2].imshow(source.astype(np.uint8))
#                 ax[0, 2].axis('off')
#                 
#                 ax[1, 0].axis('off')
#                 
#                 ax[1, 1].set_title("w1")
#                 ax[1, 1].imshow(weight1)
#                 ax[1, 1].axis('off')
#                 
#                 ax[1, 2].set_title("w2")
#                 ax[1, 2].imshow(weight2)
#                 ax[1, 2].axis('off')
#                 
#                 ax[2, 0].set_title("e*w1*w2")
#                 ax[2, 0].imshow(e * weight1 * weight2)
#                 ax[2, 0].axis('off')
#                 
#                 ax[2, 1].set_title("w1*w2")
#                 ax[2, 1].imshow(weight1 * weight2)
#                 ax[2, 1].axis('off')
#                 
#                 ax[2, 2].axis('off')
#                 plt.waitforbuttonpress()
# =============================================================================
            
            if distance < min_dist:
                min_source = source
                min_dist = distance
                min_center_x = i
    return min_center_x, ddd, fig_tmp

def SW_correspond(img1, img2, block_size, gt):
    x_search_range = 40
    y_search_range = 1
    
    img1_LAB = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB) # H, W, C
    img2_LAB = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB) # H, W, C
    img1 = img1.astype(np.float)
    img2 = img2.astype(np.float)
    img1_LAB = img1_LAB.astype(np.float)
    img2_LAB = img2_LAB.astype(np.float)
    h, w = img1.shape[0], img1.shape[1]
    
    disp_map = np.zeros((h, w))
    p_strength = proximity_strength(block_size)
    for j in tqdm.tqdm(range(block_size // 2, (h - block_size // 2))):
        for i in range(block_size // 2, (w - block_size // 2)):
            target_LAB = img1_LAB[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            target = img1[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            weight1 = similarity_strength(target_LAB, img1_LAB[j, i]) * p_strength
            
            disparity, ddd, fig_tmp = SW_block_match(i, j, target, target_LAB, img2, img2_LAB, block_size, x_search_range, y_search_range, weight1, p_strength)
            disp_map[j, i] = abs(disparity - i)
# =============================================================================
#             if (abs(disparity - i) - gt[j, i]) > 5 and i > block_size // 2 and i < w - block_size // 2 and j > block_size // 2 and j < h - block_size // 2:
#                 print(j, i, disparity)
#                 print(ddd)
#                 print(int(gt[j, i]))
#                 img2_idx = disparity # 15
#                 img1_idx = i
#                 gt_idx = int(gt[j, i]) + i
#                 print(gt_idx)
#                 fig_tmp = fig_tmp[20 + int(gt[j, i])]
#                 j_idx = j
#                 source_LAB = img2_LAB[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, img2_idx - block_size // 2 : img2_idx + block_size // 2 + 1]
#                 target_LAB = img1_LAB[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, img1_idx - block_size // 2 : img1_idx + block_size // 2 + 1]
#                 source = img2[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, img2_idx - block_size // 2 : img2_idx + block_size // 2 + 1, ::-1]
#                 source_gt = img2[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, gt_idx - block_size // 2 : gt_idx + block_size // 2 + 1, ::-1]
#                 target = img1[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, img1_idx - block_size // 2 : img1_idx + block_size // 2 + 1, ::-1]
#                 plt.figure("calculate")
#                 plt.imshow(img2[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, img2_idx - block_size // 2 : img2_idx + block_size // 2 + 1, ::-1].astype(np.uint8))
#                 plt.figure("base")
#                 plt.imshow(img1[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, img1_idx - block_size // 2 : img1_idx + block_size // 2 + 1, ::-1].astype(np.uint8))
#                 plt.figure("gt")
#                 plt.imshow(img2[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, gt_idx - block_size // 2 : gt_idx + block_size // 2 + 1, ::-1].astype(np.uint8))
#                 
#                 plt.figure("ttttt")
#                 plt.imshow(fig_tmp - img2[j_idx - block_size // 2 : j_idx + block_size // 2 + 1, gt_idx - block_size // 2 : gt_idx + block_size // 2 + 1])
#                 weight2 = similarity_strength(source_LAB, img2_LAB[j_idx, img2_idx]) * p_strength
#                 weight_gt = similarity_strength(source_gt, img2_LAB[j_idx, gt_idx]) * p_strength
#                 weight1 = similarity_strength(target_LAB, img1_LAB[j_idx, img1_idx]) * p_strength
#                 e = np.sum(abs(target - source), axis = -1)
#                 e_gt = np.sum(abs(target - source_gt), axis = -1)
#                 distance = np.sum(e * weight1 * weight2) / (np.sum(weight1 * weight2))
#                 distance_gt = np.sum(e_gt * weight1 * weight_gt) / (np.sum(weight1 * weight_gt))
#                 print("distance : ", distance)
#                 print("distance_gt : ", distance_gt)
#                 return
# =============================================================================
    return disp_map
    
def block_match(s_x, s_y, target, img, block_size, x_range, y_range):
    x_start = max(block_size // 2, s_x - x_range // 2)
    y_start = max(block_size // 2, s_y - y_range // 2)
    x_end = min(img.shape[1] - block_size // 2, s_x + x_range // 2 + 1)
    y_end = min(img.shape[0] - block_size // 2, s_y + y_range // 2 + 1)
    min_dist = np.inf
    min_center_x = None
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            source = img[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            distance = sum_square_error(target, source)
            if distance < min_dist:
                min_dist = distance
                min_center_x = i
    
    return min_center_x


def correspond(img1, img2, block_size):
    x_search_range = 40
    y_search_range = 1
    
    h, w = img1.shape[0], img1.shape[1]
    
    disp_map = np.zeros((h, w))
    
    for j in tqdm.tqdm(range(block_size // 2, (h - block_size // 2))):
        for i in range(block_size // 2, (w - block_size // 2)):
            target = img1[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            disparity = block_match(i, j, target, img2, block_size, x_search_range, y_search_range)
            disp_map[j, i] = abs(disparity - i)
    
    #disp_map = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min())
    return disp_map
    
