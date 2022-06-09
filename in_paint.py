import inpaint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import cv2


if __name__ == '__main__':
    # Load the dataset
    data_root = 'inpaint/celeba_hq_256/'
    mask_mode = 'irregular'
    dataset = inpaint.InpaintDataset(data_root,mask_mode)
    img1 = dataset[0]
    img = np.dstack((img1['cond_image'][0],img1['cond_image'][1],img1['cond_image'][2]))

    # Image.fromarray(np.asarray(img1['cond_image'])).show()
    # img_show = np.array(img1['cond_image'])
    plt.imshow(img)
    plt.show()
    # cv2.imshow(img1) cv2.waitKey(0)


