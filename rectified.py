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
    e2 = np.array([-T[1], T[0], 0]) / np.sqrt(np.sum(T[0] ** 2 + T[1] ** 2))
    e3 = np.cross(e1, e2)
    R_rect = np.array([e1, e2, e3])
    
    corner = np.array([[0, 0, 1],
                        [img.shape[1], 0, 1],
                        [0, img.shape[0], 1],
                        [img.shape[1], img.shape[0], 1]])
    
    new_corner = np.dot(R_rect, corner.T).T
    new_corner = new_corner / new_corner[:, -1:]
    
    offset_x = abs(new_corner[:, 0].min())
    offset_y = abs(new_corner[:, 1].min())
    
    offset = np.array([[1, 0, offset_x],
                       [0, 1, offset_y],
                       [0, 0, 1]])
    R_rect = np.dot(offset, R_rect)
    
    img_rect = cv2.warpPerspective(img, R_rect,
                                   (int(new_corner[:, 0].max() - new_corner[:, 0].min()),
                                    int(new_corner[:, 1].max() - new_corner[:, 1].min())))
    point = np.dot(point, R_rect)
    point = point / point[:, -1:]
    return img_rect, point, R_rect

def rect_test(img1, img2, match_keypoint1, match_keypoint2, F, T):
    import matplotlib.pyplot as plt
    
    img2_rect, point2_rect, R_rect = rectified2(img2, match_keypoint2, T)
    img1_rect, point1_rect, R_rect = rectified2(img1, match_keypoint1, T)
    # check retified result
# =============================================================================
#     lines1 = cv2.computeCorrespondEpilines(point2_rect[:, :2].reshape(-1, 1, 2), 2, F)
#     lines2 = cv2.computeCorrespondEpilines(point1_rect[:, :2].reshape(-1, 1, 2), 1, F)
# =============================================================================
    lines1 = np.matmul(point2_rect[:5], F)
    lines2 = np.matmul(F, point1_rect[:5].T).T
    lines1 = lines1.reshape(-1,3)
    lines2 = lines2.reshape(-1,3)
    img1 = draw_epipole_lines(img1_rect, lines1, point1_rect[:, :2])
    img2 = draw_epipole_lines(img2_rect, lines2, point2_rect[:, :2])
    plt.figure()
    plt.imshow(img1_rect[..., ::-1])
    plt.figure()
    plt.imshow(img2_rect[..., ::-1])
    
def draw_epipole_lines(img1, lines, pts1):
    h, w = img1.shape[0], img1.shape[1]
    for r, pt1 in zip(lines, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1, tuple(pt1.astype(np.int32)), 5, color, -1)
    return img1
    
