# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:31:12 2022

@author: Eddie
"""

import numpy as np
import matplotlib.pyplot as plt

def inv_wavelet(img_set):
    hh, hl, lh, ll = img_set
    h_odd = hh + hl
    h_even = hl - hh
    h_img = np.zeros((h_odd.shape[0] * 2, h_odd.shape[1]))
    h_img[::2] = h_even
    h_img[1::2] = h_odd
    
    l_odd = ll + lh
    l_even = ll - lh
    l_img = np.zeros((l_odd.shape[0] * 2, l_odd.shape[1]))
    l_img[::2] = l_even
    l_img[1::2] = l_odd
    
    img = np.zeros((h_img.shape[0], h_img.shape[1] * 2))
    img[:, 1::2] = l_img + h_img
    img[:, ::2] = l_img - h_img
# =============================================================================
#     plt.figure()
#     plt.imshow(img)
# =============================================================================
    return img
    

def wavelet(img):
    img = img.astype(np.float)
    # column
    even_img = img[:, ::2]
    odd_img = img[:, 1::2]
    if even_img.shape[1] != odd_img.shape[1]:
        even_img = even_img[:, :-1]
        
    h_img = even_img * -0.5 + odd_img * 0.5
    l_img = even_img * 0.5 + odd_img * 0.5
    
    # row
    h_even_img = h_img[::2]
    h_odd_img = h_img[1::2]
    l_even_img = l_img[::2]
    l_odd_img = l_img[1::2]
    if h_even_img.shape[0] != h_odd_img.shape[0]:
        h_even_img = h_even_img[:-1]
        l_even_img = l_even_img[:-1]
        
    hh_img = h_even_img * -0.5 + h_odd_img * 0.5
    hl_img = h_even_img * 0.5 + h_odd_img * 0.5
    lh_img = l_even_img * -0.5 + l_odd_img * 0.5
    ll_img = l_even_img * 0.5 + l_odd_img * 0.5
    
# =============================================================================
#     plt.figure()
#     plt.imshow(hh_img.astype(np.uint8))
#     plt.figure()
#     plt.imshow(hl_img.astype(np.uint8))
#     plt.figure()
#     plt.imshow(lh_img.astype(np.uint8))
#     plt.figure()
#     plt.imshow(ll_img.astype(np.uint8))
# =============================================================================
    return hh_img, hl_img, lh_img, ll_img

def iter_wavelet(img, iteration = 3, decomposed_layer = 2):
    H, W = img.shape[0], img.shape[1]
    img = img / 255
    img = np.pad(img, ((0, 512 - img.shape[0]), (0, 512 - img.shape[1])), mode='constant', constant_values = 0)
    result_img = img.copy()
    for i in range(iteration):
        tmp = []
        # iterative threshold
        threshold = 0.3
        # Universal Threshold
        #threshold = np.sqrt(np.log(2) * 2)
        median = np.median(img)
        mask = np.zeros_like(img)
        mask[img > median] = 1
        result_img = result_img + img * mask - result_img * mask
        for j in range(decomposed_layer):
            hh_img, hl_img, lh_img, ll_img = wavelet(img)
            tmp.append([hh_img, hl_img, lh_img, ll_img])
            print(np.sum(tmp[-1][0]))
            for k in range(3):
                # hard threshold
                tmp[-1][k][tmp[-1][k] < threshold] = 0
                # soft threshold
# =============================================================================
#                 threshold_mask = np.zeros_like(tmp[-1][k])
#                 threshold_mask[tmp[-1][k] < threshold] = 1
#                 tmp[-1][k] = np.log(threshold_mask * tmp[-1][k] + 0.1)
# =============================================================================
            threshold = threshold / 2
# =============================================================================
#             plt.figure("hh_img2{}".format(j))
#             plt.imshow(tmp[-1][0])
#             plt.figure("hl_img2{}".format(j))
#             plt.imshow(tmp[-1][1])
#             plt.figure("lh_img2{}".format(j))
#             plt.imshow(tmp[-1][2])
# =============================================================================
            result_img = inv_wavelet(tmp[-1])
        
    result_img = result_img[:H, :W]
    plt.figure("result")
    plt.imshow(result_img)
    plt.figure("result - ori")
    plt.imshow(abs(result_img - img[:H, :W]))
    return result_img
