import inpaint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import skimage.data
import cv2


if __name__ == '__main__':
    # Load the dataset
    data_root = 'inpaint/celeba_hq_256/'
    mask_mode = 'irregular'
    dataset = inpaint.InpaintDataset(data_root,mask_mode)
    img1 = dataset[0]
    img = np.dstack((img1['use_image'][0],img1['use_image'][1],img1['use_image'][2]))
    mask_paint = np.dstack((img1['mask_image'][0],img1['mask_image'][1],img1['mask_image'][2]))
    mask_inv = np.ones_like(mask_paint)-mask_paint

    # Show messed up image
    plt.imshow(img)
    plt.show()
    plt.close()

    # After Closed we reduce it
    print("Image shape :",img.shape)
    senser = inpaint.CompressedSensing(img, None, None,mask_inv)
    fin_img = senser.iterator(3)

    plt.imshow(fin_img)
    plt.show()
    plt.close()

        
