import numpy as np
from PIL import Image

import caffe

import matplotlib.pyplot as plt


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

# im_path = '/home/adrian/JointEmbedding/semanticFCN/data/sbdd/dataset/img/2008_000089.jpg'
im_path = '/home/adrian/JointEmbedding/semanticFCN/data/sbdd/dataset/img/2008_000096.jpg'
# im_path = '/home/adrian/JointEmbedding/datasets/image_embedding/syn_images_bkg_overlaid/03001627/22b40d884de52ca3387379bbd607d69e/03001627_22b40d884de52ca3387379bbd607d69e_a-08_e019_t001_d003.jpg'
# im_path = '/home/adrian/JointEmbedding/datasets/image_embedding/syn_images_bkg_overlaid/03001627/22b40d884de52ca3387379bbd607d69e/03001627_22b40d884de52ca3387379bbd607d69e_a073_e002_t-01_d003.jpg'
# im_path = '/home/adrian/JointEmbedding/datasets/image_embedding/syn_images_bkg_overlaid/03001627/22b40d884de52ca3387379bbd607d69e/03001627_22b40d884de52ca3387379bbd607d69e_a056_e017_t001_d003.jpg'
im_path = '/home/adrian/JointEmbedding/datasets/image_embedding/syn_images_bkg_overlaid/03001627/22b40d884de52ca3387379bbd607d69e/03001627_22b40d884de52ca3387379bbd607d69e_a110_e018_t-07_d003.jpg'

im = Image.open(im_path)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
in_ = in_.transpose((2, 0, 1))

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

imgplot = plt.imshow(out)
plt.show()

# DEBUG THE HEAT MAPS
probability_maps = net.blobs['upscore8'].data
max_score = probability_maps.max()
min_score = probability_maps.min()
for i in range(0, 21):
    heat_map = probability_maps[0, i, ...]
    plt.subplot(3, 7, i + 1)
    plt.imshow(heat_map, vmin=min_score, vmax=max_score)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
# plt.pause(1)

lolo = 1