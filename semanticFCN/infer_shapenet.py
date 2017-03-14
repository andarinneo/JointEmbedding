import numpy as np
from PIL import Image
import caffe
import json
import matplotlib.pyplot as plt


# LOAD TEST DATASET
# train_val_test = '../datasets/image_embedding/train_val_test.txt'
train_val_test = '/home/adrian/JointEmbedding/datasets/image_embedding/train_val_test.txt'

with open(train_val_test, 'r') as f:
    img_idxs_train, label_idxs_train, img_idxs_test, label_idxs_test, img_idxs_val, label_idxs_val = json.load(f)

im_path = img_idxs_test[1000]


# LOAD NETWORK
net = caffe.Net('/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/deploy-fcn8-atonce.prototxt', '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/snapshot/train_iter_130000.caffemodel', caffe.TEST)


# MAIN LOOP
for im_it in range(167, 209):
    # LOAD IMAGE, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    # im_path = '/home/adrian/JointEmbedding/datasets/image_embedding/syn_images_bkg_overlaid/03001627/22b40d884de52ca3387379bbd607d69e/03001627_22b40d884de52ca3387379bbd607d69e_a110_e018_t-07_d003.jpg'
    im_path = '/home/adrian/Desktop/testCases/real/chair' + str(im_it) + '.JPEG'

    im = Image.open(im_path)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = in_.transpose((2, 0, 1))

    # RESHAPE for input data (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    # PLOT IMAGE AND RESULT
    plt.subplot(3, 4, 1)
    plt.imshow(im)
    plt.subplot(3, 4, 2)
    plt.imshow(out, vmin=0, vmax=4)

    # DEBUG THE HEAT MAPS
    probability_maps = net.blobs['upscore8'].data
    max_score = probability_maps.max()
    min_score = probability_maps.min()

    for hm_it in range(0, 5):  # 21 is max
        heat_map = probability_maps[0, hm_it, ...]
        plt.subplot(3, 4, hm_it + 1 + 2)
        plt.imshow(heat_map, vmin=min_score, vmax=max_score)
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    # plt.show()
    # plt.pause(1)
    # plt.savefig('/home/adrian/Desktop/testCases/real/chair' + str(im_it) + '.png')

    plt.savefig('/home/adrian/Desktop/testCases/chair' + str(32) + '.png')


lolo = 1