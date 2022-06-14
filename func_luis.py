import numpy as np


def point_normalization(img1,img2,pts1,pts2):

    # Might be a bit computaitonal expensive
   
    c1 = (img1.shape[1]/2,img1.shape[0]/2)
    c2 = (img2.shape[1]/2,img2.shape[0]/2)
    n1 = img1.shape[0]*img1.shape[1]
    n2 = img2.shape[0]*img2.shape[1]

    # TODO: perhaps make this more efficient
    # Now calculate the variances of distances
    summ = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            summ += np.linalg.norm([j-c1[1],i-c1[0]])**2
    s1 = np.sqrt(2*n1/summ)

    summ = 0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            summ += np.linalg.norm([j-c2[1],i-c2[0]])**2
    s2 = np.sqrt(2*n2/summ)

    # At this point we have both scales.
    # Lets form our transformation matrix:
    
    T1 = np.array([
        [s1,0,-s1*c1[0]],
        [0,-s1,-s1*c1[1]],# TODO Not sure about this
        [0,0,1]
        ])
    T2 = np.array([
        [s2,0,-s2*c2[0]],
        [0,-s2,-s2*c2[1]],# TODO Not sure about this
        [0,0,1]
        ])
    
    # Now change the coordinates of the points
    npt1 = T1@pts1
    npt2 = T1@pts2
    return npt1, npt2

def get_ef_matrix(img1_sp,img2_sp):

    # Build the Matrix
    # Here we want to grab all teh coordinates of our points and normalized them

    A = np.ndarray((8,9))

    idx = 0
    for sp1, sp2  in zip(imgs_1,imgs2_sp):
        u,v = sp1.kp
        up,vp = sp1.kp
        A[idx,:] = np.array([
            u*up,v*up,up,u*vp,v*vp,vp,u,v,1
            ])
        idx +=1 
    # With Our Matrix at hand  we wantn to find our smallest eigen vector
    # Ah = 0
    u,s, vt = np.linalg.svd(A.T@A)
    f = vt[-1,:]
    F = f.resize(3,3)

    # TODO: Need to decrease teh rank here
    u,s,vt  = np.linalg.svd(F)
    s[-1,:] = 0
    F = u@s@vt

    return F
    # Remember that here we have to reduce the rank of F
 
def RANSAC_F(norm_kp1, norm_kp2, threshold = 5e-4, iterations=500):
     print("Finding the Fundamental Matrix")
     # We get a sample of 8 matching key_points
     idx = np.random.choice(norm_kp1.shape[0],8)
     samp_kp1 = norm_kp1[idx]
     samp_kp2 = norm_kp2[idx]

     # Try
     best_ctr = 0
     best_M = []
     for it in range(iterations):
        M = get_ef_matrix(samp_kp1,samp_kp2)
         
        cntr = 0
         # Test it 
        for kp1,kp2 in zip(samp_kp1,samp_kp2):
            cntr += 1 if abs(np.dot(kp1.T, M@kp2)) < threshold else 0
        if ctr > best_ctr:
            best_ctr = ctr
            best_M = M
     return M
        
def rectify(img1,img2,F):
    # First calculate epilines 
    
    # Next Calculate Epipoles





