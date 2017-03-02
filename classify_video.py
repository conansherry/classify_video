# -*- coding:utf8 -*-
#classify_video.py will classify a video using (1) singleFrame RGB model (2) singleFrame flow model (3) 0.5/0.5 singleFrame RGB/singleFrame flow fusion (4) 0.33/0.67 singleFrame RGB/singleFrame flow fusion (5) LRCN RGB model (6) LRCN flow model (7) 0.5/0.5 LRCN RGB/LRCN flow model (8) 0.33/0.67 LRCN RGB/LRCN flow model
#Before using, change RGB_video_path and flow_video_path.
#Use: classify_video.py video, where video is the video you wish to classify.  If no video is specified, the video "v_Archery_g01_c01" will be classified.

import numpy as np
import cv2
from classify_video.lrcn_classifier import lrcn_classifier
caffe_root = '/home/qin.weining/workspace/caffe-rc4-lrcn/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

test_dict = {'1': np.ones((16, 227, 227, 3), np.uint8), '2': np.ones((16, 227, 227, 3), np.uint8),
             '3': np.ones((16, 227, 227, 3), np.uint8), '4': np.ones((16, 227, 227, 3), np.uint8),
             '5': np.ones((16, 227, 227, 3), np.uint8), '6': np.ones((16, 227, 227, 3), np.uint8),
             '7': np.ones((16, 227, 227, 3), np.uint8), '8': np.ones((16, 227, 227, 3), np.uint8)}

lstm_model = 'deploy.prototxt'
RGB_lstm = 'model.caffemodel'
RGB_lstm_net = caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

while True:
    lrcn_classifier(test_dict, RGB_lstm_net)

print 'Done'
