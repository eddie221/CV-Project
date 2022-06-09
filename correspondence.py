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
    return np.exp(-delta_s / 5)

def proximity_strength(r):
    x, y = np.meshgrid(np.linspace(0, r - 1, r), np.linspace(0, r - 1, r))
    diff_x = abs(x - r // 2)
    diff_y = abs(y - r // 2)
    return np.exp(-np.sqrt((diff_x ** 2 + diff_y ** 2)) / (r / 2))


def SW_block_match(s_x, s_y, img1, img1_LAB, img2, img2_LAB, block_size, x_range, y_range, weight1, p_strength):
    x_start = max(block_size // 2, s_x - x_range // 2)
    y_start = max(block_size // 2, s_y - y_range // 2)
    x_end = min(img2_LAB.shape[1] - block_size // 2, s_x + x_range // 2 + 1)
    y_end = min(img2_LAB.shape[0] - block_size // 2, s_y + y_range // 2 + 1)
    min_center_x = None
    min_dist = np.inf
    min_source = None
    ddd = []
    for j in range(y_start, y_end):
        for i in range(x_start, x_end):
            #print(j - block_size // 2, j + block_size // 2 + 1, i - block_size // 2, i + block_size // 2 + 1)
            source_LAB = img2_LAB[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            source = img2[j - block_size // 2 : j + block_size // 2 + 1, i - block_size // 2 : i + block_size // 2 + 1]
            weight2 = similarity_strength(source_LAB, img2_LAB[j, i]) * p_strength
            e = np.sum(abs(img1 - source), axis = -1)
            e[e > 40] = 40
            distance = np.sum(e * weight1 * weight2) / (np.sum(weight1 * weight2))
            ddd.append(distance)
# =============================================================================
#             plt.figure()
#             plt.imshow(similarity_strength(source_LAB, img2_LAB[j, i]))
# =============================================================================
            if distance < min_dist:
                min_source = source
                min_dist = distance
                min_center_x = i
# =============================================================================
#     if s_x == 24 and s_y == 36:
#         print(x_start, x_end, y_start, y_end)
#         print(min_center_x, min_dist)
#         plt.figure("in")
#         plt.imshow(img2[y_start - block_size // 2 : y_start + block_size // 2 + 1, min_center_x - block_size // 2 : min_center_x + block_size // 2 + 1][..., ::-1].astype(np.uint8))
#         print(ddd)
# =============================================================================
    return min_center_x

def SW_correspond(img1, img2):
    block_size = 35
    x_search_range = 20
    y_search_range = 1
    
    img1_LAB = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB) # H, W, C
    img2_LAB = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB) # H, W, C
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
            disparity = SW_block_match(i, j, target, target_LAB, img2, img2_LAB, block_size, x_search_range, y_search_range, weight1, p_strength)
            disp_map[j, i] = abs(disparity - i)
            
            # test 
# =============================================================================
#             if abs(disparity - i) > 5:
#                 print("disparity : {}, i : {}".format(disparity, i))
#                 print(i, j)
# =============================================================================
# =============================================================================
#             if i == 24 and j == 36:
#                 img2_idx = 24 # 15
#                 img1_idx = 24
#                 source_LAB = img2_LAB[36 - block_size // 2 : 36 + block_size // 2 + 1, img2_idx - block_size // 2 : img2_idx + block_size // 2 + 1]
#                 target_LAB = img1_LAB[36 - block_size // 2 : 36 + block_size // 2 + 1, img1_idx - block_size // 2 : img1_idx + block_size // 2 + 1]
#                 source = img2[36 - block_size // 2 : 36 + block_size // 2 + 1, img2_idx - block_size // 2 : img2_idx + block_size // 2 + 1, ::-1]
#                 target = img1[36 - block_size // 2 : 36 + block_size // 2 + 1, img1_idx - block_size // 2 : img1_idx + block_size // 2 + 1, ::-1]
#                 plt.figure("out")
#                 plt.imshow(img2[36 - block_size // 2 : 36 + block_size // 2 + 1, img2_idx - block_size // 2 : img2_idx + block_size // 2 + 1, ::-1].astype(np.uint8))
#                 plt.figure("out 1")
#                 plt.imshow(img1[36 - block_size // 2 : 36 + block_size // 2 + 1, img1_idx - block_size // 2 : img1_idx + block_size // 2 + 1, ::-1].astype(np.uint8))
#                 weight2 = similarity_strength(source_LAB, img2_LAB[36, img2_idx]) * p_strength
#                 weight1 = similarity_strength(target_LAB, img1_LAB[36, img1_idx]) * p_strength
#                 e = np.sum(abs(target - source), axis = -1)
#                 plt.figure()
#                 plt.imshow(e)
#                 e[e > 40] = 40
#                 distance = np.sum(e * weight1 * weight2) / (np.sum(weight1 * weight2))
#                 print(distance)
#                 return
# # =============================================================================
# #             if i == 9:
# #                 return
# # =============================================================================
# =============================================================================
    
    #disp_map = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min())
    plt.figure()
    plt.imshow(disp_map)
    plt.imsave("./sw_result.png", disp_map, cmap='gray')
    return disp_map
    
def block_match(s_x, s_y, target, img, block_size, x_range, y_range):
    x_start = max(0, s_x - block_size)
    y_start = max(0, s_y - block_size)
    x_end = min(img.shape[1] - block_size, s_x + x_range)
    y_end = min(img.shape[0] - block_size, s_y + y_range)
    min_dist = np.inf
    min_center_x = None
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            try:
                source = img[j : j + block_size, i : i + block_size]
                distance = sum_square_error(target, source)
                #distance = 1 - cosine_similarity(target.reshape(-1), source.reshape(-1))
            except:
                print(x_end, j)
            if distance < min_dist:
                min_dist = distance
                min_center_x = i + block_size // 2 
    
    return min_center_x


def correspond(img1, img2):
    block_size = 15
    x_search_range = 20
    y_search_range = 1
    
    h, w = img1.shape[0], img1.shape[1]
    
    disp_map = np.zeros((h, w))
    
    for i in tqdm.tqdm(range(block_size, w - block_size)):
        for j in range(block_size, h - block_size):
            target_block = img1[j : j + block_size, i : i + block_size]
            disparity = block_match(i, j, target_block, img2, block_size, x_search_range, y_search_range)
            disp_map[j, i] = abs(disparity - i)
    
    disp_map = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min())
    plt.figure()
    plt.imshow(disp_map * 255)
    
