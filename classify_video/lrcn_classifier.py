import numpy as np
import os

mean_RGB = np.array([103.939, 116.779, 128.68])

def lrcn_classifier(video_dict, net):
    clip_length = 16
    key_list = []
    caffe_in = np.empty((0, 3, 227, 227))
    clip_markers_in = np.empty((0, 1, 1, 1))
    output_predictions = np.zeros((len(video_dict), 2))

    for key in video_dict:
        key_list.append(key)
        frames = video_dict[key]

        assert len(frames) == clip_length

        frames = frames - mean_RGB
        frames /= 255.0
        frames = frames.transpose((0, 3, 1, 2))
        caffe_in = np.concatenate((caffe_in, frames), axis=0)

        clip_clip_markers = np.ones((clip_length, 1, 1, 1))
        clip_clip_markers[0, :, :, :] = 0
        clip_markers_in = np.concatenate((clip_markers_in, clip_clip_markers), axis=0)

    print caffe_in.shape
    print clip_markers_in.shape
    net.blobs['data'].reshape(*(caffe_in.shape))
    net.blobs['clip_markers'].reshape(*(clip_markers_in.shape))
    out = net.forward_all(data=caffe_in, clip_markers=clip_markers_in)
    for blob in net.blobs:
        print blob, net.blobs[blob].data[...].shape
    for input_blob in net.inputs:
        print input_blob
    print out['probs'].shape
    print np.mean(out['probs'], 0).shape
    print np.argmax(np.mean(out['probs'], 0), axis=1)
    res_label = np.argmax(np.mean(out['probs'], 0), axis=1)
    output_res = {}
    index = 0
    for key in key_list:
        output_res[key] = res_label[index]
        index = index + 1
    print output_res
