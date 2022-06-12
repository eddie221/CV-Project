import numpy as np
import jax.numpy as jnp
from cr.sparse import lop

class MedianFilter():
    def __init__(self):
        pass
class Operator(object):
    def __init__(self):
        pass

class DWT_Operator(Operator):
    def __init__(self,img_shape):
        w,h,_ = img_shape
        self.op = lop.dwt2D((w,h),wavelet='haar',level=5)
        self.op = lop.jit(self.op)

    def decompose(self,img):
        r_coeff = self.op.times(img[:,:,0])
        g_coeff = self.op.times(img[:,:,1])
        b_coeff = self.op.times(img[:,:,2])
        return np.dstack([r_coeff,g_coeff,b_coeff])# Coefficients

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

    def compose(self,coeffs):
        # Do so for the three channels
        r_chann = self.op.trans(coeffs[:,:,0])
        g_chann = self.op.trans(coeffs[:,:,1])
        b_chann = self.op.trans(coeffs[:,:,2])

        return np.dstack([r_chann,g_chann,b_chann])
        # Decrease some coefficients here

class CompressedSensing(object):

    def __init__(self,image,operator,threshold,mask):
        self.opt = DWT_Operator(image.shape)
        self.image = image
        self.mask = mask
        self.it_step = 0.1

    def iterator(self,iterations):
        cur_aprox = self.image
        for it in range(iterations):
            Z = (big_X + self.image-self.mask*big_X)
            coeff = self.opt.decompose(Z)
            coeff = self.opt.threshold(coeff) #Threshold the Coefficients
            big_X = self.opt.compose(coeff)
        return big_X 

    def generate_mask(self):
        pass


