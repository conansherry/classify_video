import numpy as np
import os

mean_RGB = np.array([103.939, 116.779, 128.68])

def lrcn_classifier(video_dict, net):
    clip_length = 16
    key_list = []
    caffe_in = np.empty((0, 3, 227, 227))
    clip_markers = np.empty((0, 1, 1, 1))
    output_predictions = np.zeros((len(video_dict), 2))

    for key in video_dict:
        key_list.append(key)
        frames = video_dict[key]

        assert len(frames) == clip_length

        frames = frames - mean_RGB
        frames /= 255.0
        frames = frames.transpose((0, 3, 1, 2))
        np.concatenate((caffe_in, frames), axis=0)

        clip_clip_markers = np.ones((clip_length, 1, 1, 1))
        clip_clip_markers[0, :, :, :] = 0
        clip_markers = np.concatenate((clip_markers, clip_clip_markers), axis=0)

    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_markers))
    print out
    output_predictions = np.mean(out['probs'], 1)
    print 'out[\'probs\'].shape', out['probs'].shape
    print 'output_predictions.shape', output_predictions.shape
    print np.mean(output_predictions, 0).argmax(), output_predictions