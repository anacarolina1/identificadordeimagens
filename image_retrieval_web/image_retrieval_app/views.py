from django.shortcuts import render
from django.shortcuts import render_to_response
import numpy as np
import matplotlib.pyplot as plt
import urllib
import h5py # to save/load data files
import sys
import os
from scipy import misc # To load/save images without Caffe
from django.conf import settings
from django.core.files.storage import FileSystemStorage


images_path = 'media/'
labels = [] # Initialising labels as an empty array.

home_dir = os.getenv("HOME")

# mudar o caminho para o local do caffe no seu computador
caffe_root = os.path.join(home_dir, 'das/caffe')  


import caffe



caffe.set_mode_cpu()

model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
if not os.path.exists(labels_file):
    os.system("~/caffe/data/ilsvrc12/get_ilsvrc_aux.sh")

labels = np.loadtxt(labels_file, str, delimiter='\t')

def index(request):
    if request.method == 'POST' and request.FILES['imagem']:
        imagem = request.FILES['imagem']
        fs = FileSystemStorage()
        filename = fs.save(imagem.name, imagem)
        uploaded_file_url = fs.url(filename)

        images_vector = create_images_vector(uploaded_file_url)
        return render(request , 'resultado.html', {
            'uploaded_file_url': uploaded_file_url ,
            'images_vector' : images_vector
        })
    return render(request, 'index.html')

def predict_imageNet(image_filename):
    image = caffe.io.load_image(image_filename)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)

    # perform classification
    net.forward()

    # obtain the output probabilities
    output_prob = net.blobs['prob'].data[0]

    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:5]

    plt.imshow(image)
    plt.axis('off')

    print 'probabilities and labels:'
    predictions = zip(output_prob[top_inds], labels[top_inds]) # showing only labels (skipping the index)
    for p in predictions:
        print p

    return output_prob


def load_dataset(images_path):
    # Load/build a dataset of vectors (i.e. a big matrix) of probabilities
    # from the ImageNet ILSVRC 2012 challenge using Caffe.
    vectors_filename = os.path.join(images_path, 'vectors.h5')

    if os.path.exists(vectors_filename):
        print 'Loading image signatures (probability vectors) from ' + vectors_filename
        with h5py.File(vectors_filename, 'r') as f:
            vectors = f['vectors'][()]
            img_files = f['img_files'][()]

    else:
        # Build a list of JPG files (change if you want other image types):
        os.listdir(images_path)
        img_files = [f for f in os.listdir(images_path) if (('jpg' in f) or ('JPG') in f)]

        print 'Loading all images to the memory and pre-processing them...'

        net_data_shape = net.blobs['data'].data.shape
        train_images = np.zeros(([len(img_files)] + list(net_data_shape[1:])))

        for (f,n) in zip(img_files, range(len(img_files))):
            print '%d %s'% (n,f)
            image = caffe.io.load_image(os.path.join(images_path, f))
            train_images[n] = transformer.preprocess('data', image)

        print 'Extracting descriptor vector (classifying) for all images...'
        vectors = np.zeros((train_images.shape[0],1000))
        for n in range(0,train_images.shape[0],10): # For each batch of 10 images:
            # This block can/should be parallelised!
            print 'Processing batch %d' % n
            last_n = np.min((n+10, train_images.shape[0]))

            net.blobs['data'].data[0:last_n-n] = train_images[n:last_n]

            # perform classification
            net.forward()

            # obtain the output probabilities
            vectors[n:last_n] = net.blobs['prob'].data[0:last_n-n]

        print 'Saving descriptors and file indices to ' + vectors_filename
        with h5py.File(vectors_filename, 'w') as f:
            f.create_dataset('vectors', data=vectors)
            f.create_dataset('img_files', data=img_files)

    return vectors, img_files



def create_images_vector(uploaded_file_url):
    vectors, img_files = load_dataset(images_path)
    KNN = NearestNeighbors(Xtr=vectors, img_files=img_files, images_path=images_path, labels=labels)

    # Freeing memory:
    del vectors

    #remove a first element of uploaded_file_url string
    uploaded_file_url = list(uploaded_file_url)
    uploaded_file_url.pop(0)
    uploaded_file_url = "".join(uploaded_file_url)

    images_vector = KNN.retrieve(predict_imageNet(uploaded_file_url))
    return images_vector

 
class NearestNeighbors:
    def __init__(self, K=24, Xtr=[], images_path='Photos/', img_files=[], labels=np.empty(0)):
        # Setting defaults
        self.K = K
        self.Xtr = Xtr
        self.images_path = images_path
        self.img_files = img_files
        self.labels = labels

    def setXtr(self, Xtr):
        """ X is N x D where each row is an example."""
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = Xtr

    def setK(self, K):
        """ K is the number of samples to be retrieved for each query."""
        self.K = K

    def setImagesPath(self,images_path):
        self.images_path = images_path

    def setFilesList(self,img_files):
        self.img_files = img_files

    def setLabels(self,labels):
        self.labels = labels

    def predict(self, x):
        """ x is a test (query) sample vector of 1 x D dimensions """

        # Compare x with the training (dataset) vectors
        # using the L1 distance (sum of absolute value differences)

        distances = np.sum(np.abs(self.Xtr-x), axis = 1)

        return np.argsort(distances) # returns an array of indices of of the samples, sorted by how similar they are to x.

    def retrieve(self, x):
        nearest_neighbours = self.predict(x)
        images = []

        for n in range(self.K):
            idx = nearest_neighbours[n]
            image = (os.path.join(self.images_path, self.img_files[idx]))
            images.append('/' + image)
        return images


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
    :reurn: whole vector
    """
    im = caffe.io.load_image(img_path)
    im = caffe.io.resize_image(im, self.img_shape)
    self.net.blobs['data'].data[0] = self.transformer.preprocess('data', im)
    out = self.net.forward()
    output_vector = out['ip2'][0]
    predicted_output = np.argmax(output_vector)
    normalized_output = output_vector/float(np.max(output_vector))
    return [output_vector, normalized_output, self.label_dict[predicted_output]]
