import numpy as np
import jax.numpy as jnp
from cr.sparse import lop
import pywt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import r_
import scipy
from .utils import compare_images

class MedianFilter():
    def __init__(self):
        pass
class Operator(object):
    def __init__(self):
        pass

class DWT_Operator(Operator):
    def __init__(self,img_shape):
        w,h = img_shape
        self.op = lop.dwt2D((w,h),wavelet='haar',level=3)
        self.op = lop.jit(self.op)

    def decompose_rgb(self,img):
        r_coeff = self.op.times(img[:,:,0])
        g_coeff = self.op.times(img[:,:,1])
        b_coeff = self.op.times(img[:,:,2])
        return np.dstack([r_coeff,g_coeff,b_coeff])# Coefficients

    def decompose(self,img):
        return np.array(self.op.times(img))

    def threshold(self,coeffs):
        h,w,c = coeffs.shape
        coeffs2 = jnp.copy(coeffs)
        # TODO change that

        idx = abs(coeffs) < 0.1
        print("Pruning {} elements".format(np.sum(idx)))
        # coeffs2 = coeffs2.at[:h//4, :w//4,:].set(coeffs[:h//4, :w//4,:])
        coeffs2 = coeffs2.at[idx].set(0)

        # v,i = torch.topk(coeffs.flatten(),10,largest=False)
        # idx = np.array(np.unravel_index(i.numpy(),coeffs.shape))
        # coeffs2 = coeffs2.at[idx].set(0)

        return coeffs2

    def compose_rgb(self,coeffs):
        # Do so for the three channels
        r_chann = self.op.trans(coeffs[:,:,0])
        g_chann = self.op.trans(coeffs[:,:,1])
        b_chann = self.op.trans(coeffs[:,:,2])

        return np.dstack([r_chann,g_chann,b_chann])
        # Decrease some coefficients here
    def compose(self,coeffs):
        return np.array(self.op.trans(coeffs))

# Eddies
def edd_inv_wavelet(img_set):
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
    

def edd_wavelet(img):
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


class HaarTransform(object):
    def __init__(self,image,levels):
        self.image = image
        self.levels = levels
        self.shape = image.shape

    def exec(self):
        # Ideally Image size is a power of 2
        h,w = self.shape
        img = self.img

        # Grab Al Even Columns # even_cols = img[:,::2]

        # for j in range(h//2):
             # # Peform Average Coefficients
             # a_avg = (img[2*j,:],2*j+1]0
             # a_diff = img
             # # Perform Diffeernce
        # for i in range(w//2):
            # a_avg = image[]
def median_filter(img):
    h,w = img.shape

    # Add Padding to the Image
    ksize = 9
    p_img = np.pad(img,(ksize,ksize))
    result = np.zeros_like(img)

    for j in np.arange(ksize,h+ksize-1):
        for i in np.arange(ksize,w+ksize-1):
            # Calculate the Median in the neighborhood
            result[j-ksize,i-ksize] = np.median(p_img[
                j-ksize:j+ksize,
                i-ksize:i+ksize])

    return result

def black_mask(image):
    mask = np.copy(image)
    mask[image > 0.25] = 1
    mask[image <= 0.25] = 0
    return mask
    

class DCT_Transform(object):

    # TODO implement by yourself later
    def __init__(self,img):
        self.img = img
        self.dct_matrix = np.zeros_like(img)
    def compute(self):
        h,w = self.image.shape
        norm_factor = np.sqrt(2/h)*np.sqrt(2/w)

        for u in range(h):
            for w in range(w):
                pass # TODO implement this
                # Integration on Image
    def forward_3p(self,img):
        return scipy.fftpack.dct(scipy.fftpack.dct(img,axis=0,norm='ortho'),axis=1,norm='ortho')
    def inv_3p(self,coeff):
        return scipy.fftpack.idct(scipy.fftpack.idct(coeff,axis=0,norm='ortho'),axis=1,norm='ortho')
    def block_forward(self):
        h,w = self.img.shape
        dct = np.zeros_like(self.img)
        for i in r_[:h:8]:
            for j in r_[:w:8]:
                dct[i:(i+8),j:(j+8)] = self.forward_3p(self.img[i:(i+8),j:(j+8)])
        return dct
    def block_inv(self,coeff):
        h,w = coeff.shape
        im_dct = np.zeros_like(coeff)
        for i in r_[:h:8]:
            for j in r_[:w:8]:
                im_dct[i:(i+8),j:(j+8)] = self.inv_3p(coeff[i:(i+8),j:(j+8)] )
        return im_dct
                
        
                


def show_img(img):
    plt.figure(dpi=150)
    plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.show()
    #plt.waitforbuttonpress(0)
    plt.close()
    
# Assuming Gray scale
def median_mask(img,median,threshold):
    diff = abs(img-median)
    mask = diff > threshold
    return 1-mask

class CompressedSensing(object):

    def __init__(self,image,operator,threshold,mask):
        self.opt = DWT_Operator(image.shape)
        self.image = image
        self.mask = mask
        self.it_step = 0.1
        self.X_n = self.image
        self.X_t = 0
    def iterator(self,iterations):
        # Set your initial Guess
        # X = self.image
        X_t = np.copy(self.image)
        #X_t = np.zeros_like(self.image)
        # thresholds = [0.1,0.05,0.01,0.005]
        # thresholds = [0.1,0.05,0.01,0.005]
        thresholds = [0.45,0.20,0.15]
        
        for thresh in thresholds:
            # Calculate residual
            R = np.multiply(self.mask,(self.image-X_t))
            # Calculate the Transform
            coeffs = self.opt.decompose(X_t + R)
            # Get nth smallest
            sort_coeffs = np.sort(np.abs(coeffs.flatten()))
            nth_smallest = sort_coeffs[-int(thresh*len(sort_coeffs))]

            # Hard Threshold
            thresh_mask = abs(coeffs) < nth_smallest
            # thresh_mask = abs(coeffs) < 0.1
            print("We are pruning : {} values because they are below {} magnitude"
                    .format(np.sum(thresh_mask),nth_smallest))
            # coeffs[thresh_mask] = 0

            # Reconstruct
            X_t = self.opt.compose(coeffs)
        return X_t

    def iterator_twodict(self,L_max=0.01):
        # Set your initial Guess
        # X = self.image
        X_t = np.zeros_like(self.image)
        X_n = np.copy(self.image)
        # thresholds = [0.45,0.20,0.15]
        # delta = lamby * L_max

        img_median = np.median(self.image)
        init_thresh = img_median

        decrease_by = 0.6
        iterations = 30
        #thresholds = [init_thresh*0.6**(i) for i in range(iterations)]
        #percnt_thresh = [0.6**i for i in range(iterations)]
        thresholds = np.linspace(img_median,0.01,30)
        thresholds = np.arange(1,)

        for thresh in thresholds:
            print("At threshold {}".format(thresh))
            # PART A
            # Calculate residual
            R = np.clip(np.multiply(self.mask,(self.image-X_t-X_n)),0,1)
            print("First Residual")
            #show_img(R)
            # Calculate the Wavelet Transform 
            coeffs = self.opt.decompose(X_n + R)
            #show_img(coeffs)
            # Do thresholding
            sort_coeffs = np.sort(np.abs(coeffs.flatten()))
            nth_biggest = sort_coeffs[int((1-thresh)*len(sort_coeffs))]
            thresh_mask = abs(coeffs) < nth_biggest
            #thresh_mask = abs(coeffs) < thresh
            print("We are pruning wavelet: {} values because they are below {} magnitude"
                  .format(np.sum(thresh_mask),thresh))
            coeffs[thresh_mask] = 0
            # Reconstruct
            X_n = self.opt.compose(coeffs)
            print("Current X_n")
            show_img(X_n)


            # Calculate the Wavelet Transform 
            # PART B
            # Calculate Residual
            R = np.multiply(self.mask,(self.image-X_t-X_n))
            R = np.clip(R,0,1)
            # Calculate DCT Transform
            dct_tran = DCT_Transform(X_t+R)
            dct_coeffs = dct_tran.block_forward()
            #show_img(dct_coeffs)
            #sort_coeffs = np.sort(np.abs(dct_coeffs.flatten()))
            # nth_smallest = sort_coeffs[int(thresh*len(sort_coeffs))]
            thresh_mask = abs(dct_coeffs) < thresh
            print("We are pruning dct: {} values because they are below {} magnitude"
                    .format(np.sum(thresh_mask),thresh))
            # dct_coeffs[thresh_mask]  
            # Reconstruct
            X_t = dct_tran.block_inv(dct_coeffs)
            

        compare_images([X_t,X_n],(1,2))
        return X_t+X_n

    def iterator_not_working(self,iterations):
        # Set your initial Guess
        # X = self.image
        X_t = np.copy(self.image)
        # thresholds = [0.1,0.05,0.01,0.005]
        thresholds = [0.8,0.45,0.20,0.15]
        
        for thresh in thresholds:
            # Calculate residual
            R = np.multiply(self.mask,(self.image - X_t))
            # Calculate the Transform
            coeffs = self.opt.decompose(X_t + R)
            # Get nth smallest
            sort_coeffs = np.sort(np.abs(coeffs))
            nth_smallest = sort_coeffs.flatten()[-int(thresh*len(sort_coeffs))]

            # Hard Threshold
            # thresh_mask = abs(coeffs) < nth_smallest
            thresh_mask = abs(coeffs) < 0.1
            
            print("We are pruning : {} values because they are below {} magnitude"
                    .format(np.sum(thresh_mask),nth_smallest))
            coeffs[thresh_mask] = 0

            # Reconstruct
            X_t = self.opt.compose(coeffs)

        return X_t






    def comp_iterator(self,iterations):
        cur_aprox = self.image
        for it in range(iterations):
            # Part A
            # Calcualte Residual
            R = (self.image - self.X_t - self.X_n)
            M_R = np.multiyply(self.mask,R)

            # Part B
            Z = (big_X + self.image-self.mask*big_X)
            coeff = self.opt.decompose(Z)
            coeff = self.opt.threshold(coeff) #Threshold the Coefficients
            big_X = self.opt.compose(coeff)
        return big_X 

    def generate_mask(self):
        pass


