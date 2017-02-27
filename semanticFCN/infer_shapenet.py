import numpy as np
from PIL import Image
import caffe
import json
import matplotlib.pyplot as plt


# Load the testing images
train_val_test = '../datasets/image_embedding/train_val_test.txt'
with open(train_val_test, 'r') as f:
    img_idxs_train, label_idxs_train, img_idxs_test, label_idxs_test, img_idxs_val, label_idxs_val = json.load(f)
test = img_idxs_test


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open(test[0])
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))


# load net
net = caffe.Net('shapenet-fcn32s/deploy.prototxt', 'shapenet-fcn32s/train_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

plt.show()
imgplot = plt.imshow(out)

lolo = 1