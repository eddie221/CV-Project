# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:08:36 2022

@author: Eddie
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_match(img1, img2, topK = 250, method = "ORB"):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if method == "SIFT":
        # sift find matching point --------------------------------------------
        sift = cv2.xfeatures2d.SIFT_create()
        
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1_gray, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2_gray, None)
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
        # ---------------------------------------------------------------------
    elif method == "ORB" :
        # orb find matching point ---------------------------------------------
        orb = cv2.ORB_create(nfeatures=10000)
        
        keypoints_1, descriptors_1 = orb.detectAndCompute(img1_gray, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(img2_gray, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        # ---------------------------------------------------------------------
    
# =============================================================================
#     img1_gray = cv2.drawKeypoints(img1_gray, keypoints_1, img1_gray)
#     img2_gray = cv2.drawKeypoints(img2_gray, keypoints_2, img2_gray)
# =============================================================================
    
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

# for normalize eight point elgorithm
def normalize_point(point1, point2):
    # calculate the centroid
    centroid1_x = np.mean(point1[:, 0])
    centroid1_y = np.mean(point1[:, 1])
    centroid2_x = np.mean(point2[:, 0])
    centroid2_y = np.mean(point2[:, 1])
    
    # for average distance equal to 2 time2
# =============================================================================
#     scale1 = np.sqrt(2) * point1.shape[0] / np.sum(np.sqrt((point1[:, 0] - centroid1_x) ** 2 + (point1[:, 1] - centroid1_y) ** 2))
#     scale2 = np.sqrt(2) * point2.shape[0] / np.sum(np.sqrt((point2[:, 0] - centroid2_x) ** 2 + (point2[:, 1] - centroid2_y) ** 2))
# =============================================================================
    
    # for standard deviation equal to sqrt(2)
    scale1 = np.sqrt(2 * point1.shape[0] / np.sum(np.sqrt((point1[:, 0] - centroid1_x) ** 2 + (point1[:, 1] - centroid1_y) ** 2) ** 2))
    scale2 = np.sqrt(2 * point2.shape[0] / np.sum(np.sqrt((point2[:, 0] - centroid2_x) ** 2 + (point2[:, 1] - centroid2_y) ** 2) ** 2))

    T1 = np.array([[scale1, 0, -scale1 * centroid1_x],
                   [0, scale1, -scale1 * centroid1_y],
                   [0, 0, 1]])
    
    T2 = np.array([[scale2, 0, -scale2 * centroid2_x],
                   [0, scale2, -scale2 * centroid2_y],
                   [0, 0, 1]])

    point1 = np.dot(T1, point1.T).T
    point2 = np.dot(T2, point2.T).T
# =============================================================================
#     assert int(np.mean(np.sqrt(np.sum(((point1 - np.array([[0, 0, 1]])) ** 2), axis = 1))) * 1000) == int(np.sqrt(2) * 1000), "large error when normalize"
#     assert int(np.mean(np.sqrt(np.sum(((point2 - np.array([[0, 0, 1]])) ** 2), axis = 1))) * 1000) == int(np.sqrt(2) * 1000), "large error when normalize"
# =============================================================================
    
    return point1, point2, T1, T2
    
    
# or essential matrix if use the same camera
def solve_fundamental_matrix(point1, point2):
    point1, point2, T1, T2 = normalize_point(point1, point2)
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
    F = np.dot(T2.T, np.dot(F, T1))
    uf, vs, vft = np.linalg.svd(F)
    vs[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i][i] = vs[i]
    F = np.dot(np.dot(uf, s), vft)
    return F
    
def RANSAC_F(point1, point2, threshold = 0.005):
# =============================================================================
#     point1 = point1[:8]
#     point2 = point2[:8]
# =============================================================================
    max_count = 0
    best_F = 0
    for i in range(1000):
        count = 0
        indexes = np.random.choice(point1.shape[0], 8)
        indexes = sorted(indexes)
        F = solve_fundamental_matrix(point1[indexes], point2[indexes])
        #F = solve_fundamental_matrix(point1, point2)
        for j in range(point1.shape[0]):
            distance = abs(np.dot(point2[j].T, np.dot(F, point1[j])))
            #print(distance)
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
    #T = T / np.sum(T * T)
    if T[0] < 0:
        T = -T
    return T

def solve_rotation(F, T):
    # other method to check (from wiki)
    W_inv = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1.]])
    u, vs, vt = np.linalg.svd(F)
    R = np.dot(np.dot(u, W_inv), vt)
    if np.linalg.det(R) < 0:
        print("R reflect")
        R = np.dot(R, np.array([[-1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]))
# =============================================================================
#     print(R / R[-1, -1])
#     print(np.linalg.det(R))
#     print(np.cross(F[:, 0], T))
#     R = np.array([np.cross(F[:, 0], T) + np.cross(F[:, 1], F[:, 2]), 
#                   np.cross(F[:, 1], T) + np.cross(F[:, 2], F[:, 0]),
#                   np.cross(F[:, 2], T) + np.cross(F[:, 0], F[:, 1])])
#     R = R / R[-1, -1]
#     print(R / R[-1, -1])
#     print(np.dot(R[:, 0], R[:, 1]))
#     print(np.linalg.det(R))
# =============================================================================
    return R

