# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:41:42 2022

@author: Eddie
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm

def sum_square_error(target, source):
    return np.sum((target - source) ** 2)

def cosine_similarity(target, source):
    return np.dot(target, source) / (np.sqrt(np.sum(target ** 2)) * np.sqrt(np.sum(source ** 2)) + 1e-8)

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
    
