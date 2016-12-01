import sys
import numpy as np
# need to change
# caffe_root = '/install_dir/caffe/'

import caffe
import caffe.proto.caffe_pb2
import numpy as np
import matplotlib.pyplot as plt

import urllib 
import h5py # to save/load data files

import sys
import os

from scipy import misc # To load/save images without Caffe

labels = [] # Initialising labels as an empty array.

caffe.set_mode_cpu()
# The caffe module needs to be on the Python path;
# we'll add it here explicitly.
home_dir = os.getenv("HOME")
caffe_root = os.path.join(home_dir, 'das/caffe')  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe

caffe.set_mode_cpu()

model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

class classifier(object):
    """docstring for Classifier"""
    def __init__(self, deployment_model, caffe_model, img_mean):

        # load ImageNet labels
        labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
        if not os.path.exists(labels_file):
            os.system("~/caffe/data/ilsvrc12/get_ilsvrc_aux.sh")
            
        labels = np.loadtxt(labels_file, str, delimiter='\t')

        self.img_shape = (300, 200)
        # instruction to use gpu zero
        # caffe.set_device(0)
        # set caffe mode cpu or gpu. check model_solver.prototxt.
        caffe.set_mode_cpu()
        self.net = caffe.Net(deployment_model, caffe_model, caffe.TEST)
        # getshape of the input "data" layer
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        # change to h*w*c to c*h*w
        self.transformer.set_transpose('data', (2, 0, 1))
        # mean pixel, set mean.npy
        self.transformer.set_mean('data', np.load(img_mean))

    def classify(self, img_path):
        """
        classify input image

        :param img_path: The path to input image.
        :type img_path: str
        :return: predicted label (string)
        """
        im = caffe.io.load_image(img_path)
        im = caffe.io.resize_image(im, self.img_shape)
        self.net.blobs['data'].data[0] = self.transformer.preprocess('data', im)
        out = self.net.forward()
        predicted_output = np.argmax(out['ip2'][0])
        return self.label_dict[predicted_output]

    def get_predicition_result(self, img_path):
        """
        compute the output vector in the last layer

        :param img_path: The path to input image.
        :type img_path: str
        :return: whole vector
        """
        im = caffe.io.load_image(img_path)
        im = caffe.io.resize_image(im, self.img_shape)
        self.net.blobs['data'].data[0] = self.transformer.preprocess('data', im)
        out = self.net.forward()
        output_vector = out['ip2'][0]
        predicted_output = np.argmax(output_vector)
        normalized_output = output_vector/float(np.max(output_vector))
        return [output_vector, normalized_output, self.label_dict[predicted_output]]


if __name__ == '__main__':
    pass
