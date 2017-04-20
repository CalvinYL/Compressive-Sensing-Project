#!/usr/bin/env python

import cv2
import glob
import numpy as np
import skimage
import skimage.segmentation as segmentation
import matplotlib.pyplot as plt
import imageio
from hou_saliency import Saliency
import time
import random
import math
import scipy.ndimage.morphology as morph


#modify to change the threshold for saliency value 0 - 1
#alpha = 0.1
start = time.clock()
test_images = []
#used for video processing
cap = cv2.VideoCapture("clip_3.avi")
while True:
    flag, frame = cap.read()
    if flag:
        test_images.append(frame)
    else:
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        break

#used for webcam feed
#cv2.namedWindow("preview")
#vc = cv2.VideoCapture(0)

#if vc.isOpened(): # try to get the first frame
#    rval, frame = vc.read()
#else:
#    rval = False

#test_images = ["109256.jpg"]
#for img in glob.glob("Original_img/*.jpg"):
#    test_images.append(img)

output_video = []
count = 0
for i in test_images:#test_images:#
#webcam option
#while rval:
    #rval, frame = vc.read()
    img = i#cv2.resize(frame,(500,500))#cv2.imread(i)#cv2.resize(frame,(500,500)) # #cv_img[i*3]
    out_img = np.zeros_like(img)
    imgsize  = img.shape
    img_width  = imgsize[1]
    img_height = imgsize[0]
    sal_ = Saliency(img, use_numpy_fft=True, gauss_kernel=(3, 3)).get_proto_objects_map()
    sal = cv2.resize(sal_,(img_width,img_height))
    print("Saliency Computed at time ",time.clock() - start)

    cv2.imshow("preview", sal)
    #key = cv2.waitKey(100)
    #if key == 27: # exit on ESC
    #    break
    name = "video_out/test%d.png" % (count)
    cv2.imwrite(name,sal)
    #print("Finished at time ",time.clock() - start)
    output_video.append(sal)
    count+=1
#cv2.destroyWindow("preview")
imageio.mimsave('video_test.gif', output_video)
