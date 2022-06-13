import inpaint
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import skimage.data
import cv2


if __name__ == '__main__':
    # Load the dataset
    data_root = 'inpaint/celeba_hq_256/'
    mask_mode = 'irregular'
    single_image = './inpaint/images/easier.png'
    #single_image = './inpaint/images/sw_result.png'

    # dataset = inpaint.InpaintDataset(data_root,mask_mode)
    # img1 = dataset[0]
    # img = np.dstack((img1['use_image'][0],img1['use_image'][1],img1['use_image'][2]))
    # img = img1['use_image'].sum(axis=0)
    # img /= 3 # make it into black and white
    # img = img.detach().numpy()

    # Our Images
    img = Image.open(single_image)#.convert('RGB')
    img = np.array(ImageOps.grayscale(img),dtype=np.float64)
    img = np.divide(img,255)
    print("Image shape {}".format(img.shape))
    median = inpaint.median_filter(img)
    #mask = inpaint.median_mask(img,median,0.1)
    mask = inpaint.black_mask(img)


    # Show messed up image
    inpaint.show_img(img)
    inpaint.show_img(img*mask)
    
    print("Mask")
    inpaint.show_img(mask)

    # After Closed we reduce it
    print("Image shape :",img.shape)
    senser = inpaint.CompressedSensing(img, None, None,mask)
    fin_img = senser.iterator_twodict()

    inpaint.compare_images([img,fin_img],(1,2))
        
