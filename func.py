# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:08:36 2022

@author: Eddie
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_match(img1, img2, topK = 200):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # sift find matching point ------------------------------------------------
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_gray, None)
    
    img1_gray = cv2.drawKeypoints(img1_gray, keypoints_1, img1_gray)
    img2_gray = cv2.drawKeypoints(img2_gray, keypoints_2, img2_gray)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:topK]
    
    match_keypoint1 = []
    match_keypoint2 = []
    for match in matches:
        match_keypoint1.append(keypoints_1[match.queryIdx].pt)
        match_keypoint2.append(keypoints_2[match.trainIdx].pt)
        
    match_keypoint1 = np.array(match_keypoint1)
    match_keypoint2 = np.array(match_keypoint2)
    match_keypoint1 = np.append(match_keypoint1, np.ones((match_keypoint1.shape[0], 1)), 1)
    match_keypoint2 = np.append(match_keypoint2, np.ones((match_keypoint2.shape[0], 1)), 1)
    
# =============================================================================
#     img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches, None, flags = 2)
#     plt.figure()
#     plt.imshow(img3[..., ::-1])
# =============================================================================
    # -------------------------------------------------------------------------
    return match_keypoint1, match_keypoint2

# or essential matrix if use the same camera
def solve_fundamental_matrix(point1, point2):
    A = np.ones((point1.shape[0], 9))
    A[:, 0] = point1[:, 0] * point2[:, 0]
    A[:, 1] = point1[:, 0] * point2[:, 1]
    A[:, 2] = point1[:, 0] * point2[:, 2]
    A[:, 3] = point1[:, 1] * point2[:, 0]
    A[:, 4] = point1[:, 1] * point2[:, 1]
    A[:, 5] = point1[:, 1] * point2[:, 2]
    A[:, 6] = point1[:, 2] * point2[:, 0]
    A[:, 7] = point1[:, 2] * point2[:, 1]
    A[:, 8] = point1[:, 2] * point2[:, 2]
    u, s, vt = np.linalg.svd(np.matmul(A.T, A))
    # find the eigenvector correspond to smallest eigenvalue
    F = vt[-1]
    #h = h / np.sum(h ** 2)
    F = F.reshape(3, 3)
    # decrease the rank of F
    uf, vs, vft = np.linalg.svd(F)
    vs[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i][i] = vs[i]
    
    F = np.dot(np.dot(uf, s), vft)
    return F
    
def RANSAC_F(point1, point2):
    threshold = 0.05
    max_count = 0
    best_F = 0
    for i in range(1000):
        count = 0
        indexes = np.random.choice(point1.shape[0], 8)
        F = solve_fundamental_matrix(point1[indexes], point2[indexes])
        
        for j in range(point1.shape[0]):
            distance = abs(np.dot(point2[j].T, np.dot(F, point1[j])))
            if distance < threshold:
                count += 1
                
        if count > max_count:
            max_count = count
            print("max_count : ", max_count)
            best_F = F
    return best_F

def solve_translation(F, point1, point2):
    # other method to check (from wiki)
# =============================================================================
#     W = np.array([[0, -1, 0],
#                   [1, 0, 0],
#                   [0, 0, 1.]])
#     u, vs, vt = np.linalg.svd(F)
#     s = np.zeros((3,3))
#     for i in range(2):
#         s[i][i] = vs[1]
#     t = np.dot(np.dot(np.dot(u, W), s), u.T)
#     x = t[2, 1]
#     y = t[0, 2]
#     z = t[1, 0]
#     t = np.array([x, y, z])
#     print(t / t[-1])
# =============================================================================
    u, s, vt = np.linalg.svd(np.matmul(F, F.T))
    T = vt[-1]
    T = T / np.sum(T * T)
    if T[0] < 0:
        T = -T
    return T

def solve_rotation(F, T):
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1.]])
    u, vs, vt = np.linalg.svd(F)
    R = np.dot(np.dot(u, np.linalg.inv(W)), u)
    return R

