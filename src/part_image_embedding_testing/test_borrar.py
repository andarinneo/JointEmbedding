import os
import sys
import lmdb
import shutil
import datetime
import numpy as np
import random

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_caffe import *

sys.path.append(os.path.join(g_caffe_install_path, 'python'))
import caffe


part_id = 1
print 'My training'
# My training
g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part' + str(part_id) + '.txt'  # Is correct
image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn.prototxt'  # Is correct
image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_40000.caffemodel'


train_val_split = [int(line.strip()) for line in open(g_syn_images_train_val_split, 'r')]
imageid2shapeid = [int(line.strip()) for line in open(g_syn_images_imageid2shapeid, 'r')]

print 'Loading shape embedding space from %s...'%(g_shape_embedding_space_file_txt)
shape_embedding_space = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt, 'r')]

image_test = []
image_test.append('/home/adrian/Desktop/testCases/training/chair6.jpg')  # 35, 38, 42, 55, 72, 81, 35, 01-06
image_test.append('/home/adrian/Desktop/testCases/training/chair7.jpg')  # 35, 38, 42, 55, 72, 81, 35, 01-06
image_test.append('/home/adrian/Desktop/testCases/training/chair8.jpg')  # 35, 38, 42, 55, 72, 81, 35, 01-06
image_test.append('/home/adrian/Desktop/testCases/training/chair9.jpg')  # 35, 38, 42, 55, 72, 81, 35, 01-06

image_embedding = []
for i in range(0, 4):
    print 'Image embedding for %s is:' % (image_test[i])
    image_embedding_array = extract_cnn_features(img_filelist=image_test[i],
                                                 img_root='/',
                                                 prototxt=image_embedding_prototxt,
                                                 caffemodel=image_embedding_caffemodel,
                                                 feat_name='image_embedding_part' + str(part_id),
                                                 caffe_path=g_caffe_install_path,
                                                 mean_file=g_mean_file)[0]

    image_embedding.append(image_embedding_array)

    assert(image_embedding[i].size == shape_embedding_space[0].size)


for i in range(0, 4):
    print 'Computing distances and ranking...'
    sorted_distances = sorted([(math.sqrt(sum((image_embedding[i]-shape_embedding)**2)), idx) for idx, shape_embedding in enumerate(shape_embedding_space)])
    print sorted_distances[0:5]



caffe_dist_v = []
euclidean_dist_v = []

manifold_baricenter = np.zeros(128)
counter = 0
for idx, shape_embedding in enumerate(shape_embedding_space):
    manifold_baricenter = manifold_baricenter + shape_embedding
    counter += 1
manifold_baricenter = manifold_baricenter/counter


for idx, shape_embedding in enumerate(shape_embedding_space):
    diff = manifold_baricenter - shape_embedding
    caffe_dist = np.dot(diff, diff)/(1*2)  # N (batch_size) in our case is always 1 because we only use one sample
    euclidean_dist = math.sqrt(sum((diff)**2))


    aux2 = np.multiply(diff, diff)
    aux3 = np.sum(aux2)
    loss = aux3/(1*2)  # N (batch_size) in our case is always 1 because we only use one sample

    caffe_dist_v.append(caffe_dist)
    euclidean_dist_v.append(euclidean_dist)

mean_caffe = np.mean(np.asarray(caffe_dist_v))
mean_euclidean = np.mean(np.asarray(euclidean_dist_v))

lolo = 1



# Gt Training
# batch_num: 1
# 16:03:01.970313 - batch:  0 of 1 idx range:[ 0 1 ]
# Computing distances and ranking...
# [(14.109786352940953, 565), (14.152578419341754, 2018), (14.338463146924047, 3745), (14.362117470262337, 3942), (14.46414663210452, 527)]
# Computing distances and ranking...
# [(12.182813389513944, 403), (12.246827469867375, 1794), (12.464781034635063, 3745), (12.664361486227767, 1121), (12.693991774934899, 3525)]
# Computing distances and ranking...
# [(13.773469843095816, 5177), (14.08900196035698, 5831), (14.404621238938676, 4380), (14.62395390414804, 2510), (14.69298001844357, 2245)]
# Computing distances and ranking...
# [(13.113149539400489, 3942), (13.426089022071434, 2018), (13.57569006167885, 6349), (13.587307812171556, 1794), (13.722692947615482, 4470)]


# My Training
# batch_num: 1
# 16:02:10.053308 - batch:  0 of 1 idx range:[ 0 1 ]
# Computing distances and ranking...
# [(14.963129021453245, 3745), (15.480205402101115, 1419), (15.515905365652323, 788), (16.258761087412537, 106), (16.614017313048546, 6321)]
# Computing distances and ranking...
# [(14.255580833728652, 3745), (14.607153052331801, 148), (14.814308261675235, 1794), (14.851978453473558, 527), (14.9967237030103, 5653)]
# Computing distances and ranking...
# [(15.293925546645253, 182), (15.457424987809269, 3745), (16.448344612508123, 4268), (16.56053986627816, 1477), (16.725231670361428, 1412)]
# Computing distances and ranking...
# [(14.829630936548462, 3745), (15.128706754483318, 788), (15.154357719244878, 1419), (15.37455544518379, 1794), (15.766554025257072, 106)]