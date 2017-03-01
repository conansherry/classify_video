# -*- coding:utf8 -*-
#classify_video.py will classify a video using (1) singleFrame RGB model (2) singleFrame flow model (3) 0.5/0.5 singleFrame RGB/singleFrame flow fusion (4) 0.33/0.67 singleFrame RGB/singleFrame flow fusion (5) LRCN RGB model (6) LRCN flow model (7) 0.5/0.5 LRCN RGB/LRCN flow model (8) 0.33/0.67 LRCN RGB/LRCN flow model
#Before using, change RGB_video_path and flow_video_path.
#Use: classify_video.py video, where video is the video you wish to classify.  If no video is specified, the video "v_Archery_g01_c01" will be classified.

import numpy as np
import glob
caffe_root = '/home/qin.weining/workspace/caffe-rc4-lrcn/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import pickle

def initialize_transformer(image_mean, is_flow):
  shape = (10*16, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', is_flow)
  return transformer


ucf_mean_RGB = np.zeros((3,1,1))
ucf_mean_flow = np.zeros((3,1,1))
ucf_mean_flow[:,:,:] = 128
ucf_mean_RGB[0,:,:] = 103.939
ucf_mean_RGB[1,:,:] = 116.779
ucf_mean_RGB[2,:,:] = 128.68

transformer_RGB = initialize_transformer(ucf_mean_RGB, False)
transformer_flow = initialize_transformer(ucf_mean_flow,True)

lstm_model = 'deploy.prototxt'
RGB_lstm = 'model.caffemodel'
RGB_lstm_net = caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

print 'Done'