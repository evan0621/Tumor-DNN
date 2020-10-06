import os
import cv2
import numpy as np

imgdir = '/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/train_seg_deeplab/'
savedir = '/home/lab70636/Datasets/Ultrasound_tumor/good_good_data/train_seg_cont/'
imglist = os.listdir(imgdir)
for imgn in imglist:
    img = cv2.imread(imgdir + imgn, 0)
    img = cv2.equalizeHist(img)
    # a = 2
    # img = float(a) * img
    # img[img > 255] = 255
    # img = np.round(img)
    # img = img.astype(np.uint8)
    cv2.imwrite(savedir + imgn, img)
