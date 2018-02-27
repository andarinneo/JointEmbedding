import numpy as np
from PIL import Image
import caffe
import json
import matplotlib.pyplot as plt


# LOAD TEST DATASET
# train_val_test = '../datasets/image_embedding/train_val_test.txt'
train_val_test = '/home/adrian/JointEmbedding/datasets/image_embedding/train_val_test.txt'


image_embedding_prototxt = '/home/adrian/JointEmbedding/semanticFCN/shapenet-manifold/train-manifold-alexnet.prototxt'
image_embedding_caffemodel = '/home/adrian/JointEmbedding/semanticFCN/shapenet-manifold/snapshot/initial_stacked.caffemodel'


# LOAD NETWORK
net = caffe.Net(image_embedding_prototxt, image_embedding_caffemodel, caffe.TEST)


# MAIN LOOP
for im_it in range(1, 210):  # 1-209
    # LOAD IMAGE, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    # im_path = '/home/adrian/JointEmbedding/datasets/image_embedding/syn_images_bkg_overlaid/03001627/22b40d884de52ca3387379bbd607d69e/03001627_22b40d884de52ca3387379bbd607d69e_a110_e018_t-07_d003.jpg'
    # im_path = '/home/adrian/Desktop/embedding_partial_results/testCases/real/chair' + str(im_it) + '.JPEG'
    path = '/home/adrian/Desktop/ExactPartMatchChairsDataset/part2+part3/case' + str(im_it) + '/'
    im_path = path + 'part2.jpg'

    im = Image.open(im_path)
    im2 = im.convert('RGB')
    in_ = np.array(im2, dtype=np.float32)
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
    probability_maps = net.blobs['score'].data
    max_score = probability_maps.max()
    min_score = probability_maps.min()

    for hm_it in range(0, 5):  # 21 is max
        heat_map = probability_maps[0, hm_it, ...]
        plt.subplot(3, 4, hm_it + 1 + 2)
        plt.imshow(heat_map, vmin=min_score, vmax=max_score)

    # Save figure
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    plt.pause(1)
    # plt.savefig('/home/adrian/Desktop/testCases/real/chair' + str(im_it) + '.png')
    plt.savefig(path + '/chair' + str(im_it) + '.png')


lolo = 1