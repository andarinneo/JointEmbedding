#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt

from global_variables import *
from utilities_caffe import *
from scipy.spatial import distance_matrix


def run_top_k_results(shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feature_name, top_k_values):
    path = '/home/adrian/JointEmbedding/src/experiments/ExactMatchChairsDataset'
    imgname_filelist_file = path + '/' + 'exact_match_chairs_img_filelist.txt'
    new_imgname_filelist_file = path + '/' + 'exact_match_chairs_img_filelist_modified.txt'

    # ----------------          Load dataset lists and data          ----------------
    f = open(new_imgname_filelist_file, 'w+')
    imgname_list = []
    for line in open(imgname_filelist_file, 'r'):
        imgname = line.strip().replace('/orions3-zfs/projects/rqi/Dataset/ExactMatchChairsDataset/', '')
        imgname_list.append(imgname)
        f.write(imgname + '\n')
    f.close()
    n_imgs = len(imgname_list)
    print n_imgs, 'images!'

    shape_list_file = path + '/' + 'exact_match_chairs_shape_modelIds_0to6776.txt'
    shape_list = [int(line.strip()) for line in open(shape_list_file, 'r')]
    n_shapes = len(shape_list)
    print n_shapes, 'images!'

    gt_img2shape_file = path + '/' + 'exact_match_chairs_img_modelIds_0to6776.txt'
    gt_img2shape = [int(line.strip()) for line in open(gt_img2shape_file, 'r')]

    # ----------------          Load manifold and 105 models values          ----------------
    print 'Loading shape embedding space from %s...' % (shape_embedding_space_file_txt)
    shape_embedding_space = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(shape_embedding_space_file_txt, 'r')]

    shape_list_np = np.asarray(shape_list)
    shape_embedding_space_np = np.asarray(shape_embedding_space)

    shape_embedding_array = shape_embedding_space_np[shape_list_np, :]

    # ----------------          Evaluate image in network          ----------------
    image_embedding_array = extract_cnn_features(img_filelist=new_imgname_filelist_file,
                                                 img_root=path,
                                                 prototxt=image_embedding_prototxt,
                                                 caffemodel=image_embedding_caffemodel,
                                                 feat_name=feature_name,
                                                 caffe_path=g_caffe_install_path,
                                                 mean_file=g_mean_file)
    image_embedding_array = np.asarray(image_embedding_array)

    # ----------------          Find distance to 105 3D model in manifold space          ----------------
    results_list = []
    for top_k in top_k_values:
        dist_mat = distance_matrix(image_embedding_array, shape_embedding_array)
        results_single_manifold = [dist_mat[i, :].argsort()[:top_k] for i in range(n_imgs)]

        bool_vec = []
        for i in range(n_imgs):
            aux_vec = shape_list_np[results_single_manifold[:][i]]
            bool_val = bool(aux_vec[np.where(aux_vec == gt_img2shape[i])].shape[0])
            bool_vec.append(bool_val)

        n_correct = sum(bool_vec)
        results_list.append((100 * n_correct / n_imgs))
        print 'RESULTS:', n_correct, 'correct matches using top k:', top_k, ', percentage:', (100 * n_correct / n_imgs), '%'

    return results_list



top_k_values = range(1,32,2)

# GT Single Manifold
g_shape_embedding_space_file_txt = '/home/adrian/Desktop/03001627/shape_embedding_space_03001627.txt'  # Is correct
image_embedding_prototxt = '/home/adrian/Desktop/03001627/image_embedding_03001627.prototxt'  # Is correct
image_embedding_caffemodel = '/home/adrian/Desktop/03001627/image_embedding_03001627.caffemodel'
feat_name = 'image_embedding'

gt_results = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Single Manifold
g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/shape_embedding_space_03001627(myTraining).txt'  # Is correct
image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/image_embedding_rcnn.prototxt'  # Is correct
image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_20000(itWorks).caffemodel'
feat_name = 'image_embedding'

single_results = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Part Manifold (only 1 part)
g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold.prototxt'  # Is correct
image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_400000.caffemodel'
feat_name = 'image_embedding_part1'

single_part_results = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


plt.xlabel('top k retrieval')
plt.ylabel('Probability')
plt.title('ExactMatch Dataset results')
plt.plot(top_k_values, gt_results, 'b', label='GT Whole Chair')
plt.plot(top_k_values, single_results, 'g', label='My Whole Chair')
plt.plot(top_k_values, single_part_results, 'r', label='Single Part (1)')
plt.legend(loc=4)

ax = plt.gca()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Customize the major grid
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth='0.4 ', color='red')  # Customize the minor grid
plt.axis([0, 32, 0, 100])
plt.show()

lolo = 1


