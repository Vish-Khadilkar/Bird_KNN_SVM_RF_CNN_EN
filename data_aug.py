##Data Augmentation Code
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import os
import string

CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'spec')))]

for c in CLASSES:
    os.mkdir(os.path.join(cfg.TRAINSET_PATH, 'spec_aug', c))
    ifiles = [f for f in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'spec', c)))]
    for img_read in ifiles:
        if img_read.endswith(".png"):
            readpath = os.path.join(cfg.TRAINSET_PATH, 'spec', c, img_read)
            writepath = os.path.join(cfg.TRAINSET_PATH, 'spec_aug', c, img_read)
            writepath_vr = cfg.TRAINSET_PATH + '/' + 'spec_aug' + '/' + c + '/' + 'vr_' + img_read
            writepath_ns = cfg.TRAINSET_PATH + '/' + 'spec_aug' + '/' + c + '/' + 'ns_' + img_read
            print(readpath)
            #print(img_read)
            #print(readpath)
            #print(writepath)
            #img = cv2.imread(readpath + img_read)
            #img = np.array(img)
            #plt.imshow(np.array(cv2.imread(readpath)))
            #plt.show()
            cv2.imwrite(writepath, np.array(cv2.imread(readpath)))
            #print(np.shape(np.array(cv2.imread(readpath)).flatten('F').reshape(8192, -1)))
            cv2.imwrite(writepath_vr, np.roll(np.array(cv2.imread(readpath)), 5))
            cv2.imwrite(writepath_ns, np.roll(np.array(cv2.imread(readpath)), 50))
            img = []
            img = np.array(cv2.imread(readpath)) + np.random.normal(0.0,0.05,0.5)
            # v roll
            #img_v_roll = np.roll(np.array(cv2.imread(readpath)), 1)
            #cv2.imwrite('/home/vish/Pictures/im_v_roll.png', img_v_roll)

#plt.imshow(img)

# Flipping images with Numpy
#flipped_img = np.fliplr(img)
#plt.imshow(flipped_img)
#plt.imshow(img_v_roll)



#plt.show()
