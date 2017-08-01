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


def run_top_k_blended_results(shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feature_name_part1,
                              shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feature_name_part2,
                              shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feature_name_part3,
                              shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feature_name_part4,
                              top_k_values):

    path = '/home/adrian/JointEmbedding/src/experiments/ExactPartMatchChairsDataset'

    # ----------------          Load manifold and N model values          ----------------
    print 'Loading shape embedding space from %s...' % (shape_embedding_space_file_txt_part1)
    shape_embedding_space_part1 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(shape_embedding_space_file_txt_part1, 'r')]
    shape_embedding_space_part1_np = np.asarray(shape_embedding_space_part1)

    print 'Loading shape embedding space from %s...' % (shape_embedding_space_file_txt_part2)
    shape_embedding_space_part2 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(shape_embedding_space_file_txt_part2, 'r')]
    shape_embedding_space_part2_np = np.asarray(shape_embedding_space_part2)

    print 'Loading shape embedding space from %s...' % (shape_embedding_space_file_txt_part3)
    shape_embedding_space_part3 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(shape_embedding_space_file_txt_part3, 'r')]
    shape_embedding_space_part3_np = np.asarray(shape_embedding_space_part3)

    print 'Loading shape embedding space from %s...' % (shape_embedding_space_file_txt_part4)
    shape_embedding_space_part4 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(shape_embedding_space_file_txt_part4, 'r')]
    shape_embedding_space_part4_np = np.asarray(shape_embedding_space_part4)

    shape_embedding_space_np = np.zeros([4, shape_embedding_space_part1_np.shape[0], shape_embedding_space_part1_np.shape[1]])
    shape_embedding_space_np[0, ...] = shape_embedding_space_part1_np
    shape_embedding_space_np[1, ...] = shape_embedding_space_part2_np
    shape_embedding_space_np[2, ...] = shape_embedding_space_part3_np
    shape_embedding_space_np[3, ...] = shape_embedding_space_part4_np

    feature_name = [feature_name_part1, feature_name_part2, feature_name_part3, feature_name_part4]
    image_embedding_prototxt = [image_embedding_prototxt_part1, image_embedding_prototxt_part2, image_embedding_prototxt_part3, image_embedding_prototxt_part4]
    image_embedding_caffemodel = [image_embedding_caffemodel_part1, image_embedding_caffemodel_part2, image_embedding_caffemodel_part3, image_embedding_caffemodel_part4]


    top_k_mat = []
    n_cases_list = []
    n_experiments = 1  # 5
    for experiment in range(n_experiments):
        # ----------------          Load manifold and N model values          ----------------
        if experiment == 0:
            partA_id = 1
            partB_id = 2
            n_cases = 60
        elif experiment == 1:
            partA_id = 1
            partB_id = 3
            n_cases = 40
        elif experiment == 2:
            partA_id = 2
            partB_id = 3
            n_cases = 164
        elif experiment == 3:
            partA_id = 1
            partB_id = 4
            n_cases = 30
        elif experiment == 4:
            partA_id = 2
            partB_id = 4
            n_cases = 56

        subpath = path + '/' + 'part' + str(partA_id) + '+part' + str(partB_id)

        imgname_filelist_partA = subpath + '/' + 'partA.txt'
        imgname_filelist_partB = subpath + '/' + 'partB.txt'

        n_cases_list.append(n_cases)

        # ----------------          Load dataset lists and data          ----------------
        shape_list_file = path + '/' + 'exact_part_match_chairs_shape_modelIds_0to6776.txt'
        shape_list = [int(line.strip()) for line in open(shape_list_file, 'r')]
        shape_list_np = np.asarray(shape_list)
        n_shapes = len(shape_list)
        print n_shapes, 'shapes!'

        gt_case2shape_file = subpath + '/' + 'exact_part_match_chairs_case2modelIds_0to6776.txt'
        gt_case2shape = [int(line.strip()) for line in open(gt_case2shape_file, 'r')]


        # ----------------          Obtain manifold coordinates for shapes part A         ----------------
        shape_embedding_array_A = shape_embedding_space_np[partA_id-1, shape_list_np, :]

        # Evaluate image in network for part A
        image_embedding_array_A = extract_cnn_features(img_filelist=imgname_filelist_partA,
                                                       img_root=subpath,
                                                       prototxt=image_embedding_prototxt[partA_id-1],
                                                       caffemodel=image_embedding_caffemodel[partA_id-1],
                                                       feat_name=feature_name[partA_id-1],
                                                       caffe_path=g_caffe_install_path,
                                                       mean_file=g_mean_file)
        image_embedding_array_A = np.asarray(image_embedding_array_A)

        # Compute distances between shape and estimation
        dist_mat_A = distance_matrix(image_embedding_array_A, shape_embedding_array_A)


        # ----------------          Obtain manifold coordinates for shapes part B         ----------------
        shape_embedding_array_B = shape_embedding_space_np[partB_id-1, shape_list_np, :]

        # Evaluate image in network for part B
        image_embedding_array_B = extract_cnn_features(img_filelist=imgname_filelist_partB,
                                                       img_root=subpath,
                                                       prototxt=image_embedding_prototxt[partB_id-1],
                                                       caffemodel=image_embedding_caffemodel[partB_id-1],
                                                       feat_name=feature_name[partB_id-1],
                                                       caffe_path=g_caffe_install_path,
                                                       mean_file=g_mean_file)
        image_embedding_array_B = np.asarray(image_embedding_array_B)

        # Compute distances between shape and estimation
        dist_mat_B = distance_matrix(image_embedding_array_B, shape_embedding_array_B)


        # ----------------          blend parts by minimizing the combined distances         ----------------
        blended_mat = dist_mat_A + dist_mat_B

        top_k_list = []
        for top_k in top_k_values:
            results_blended = np.asarray([blended_mat[i, :].argsort()[:top_k] for i in range(n_cases)])

            bool_vec = []
            for i in range(n_cases):
                aux_vec = shape_list_np[results_blended[:][i]]
                bool_val = bool(aux_vec[np.where(aux_vec == gt_case2shape[i])].shape[0])
                bool_vec.append(bool_val)

            n_correct = sum(bool_vec)
            top_k_list.append((100 * n_correct / n_cases))
            print 'RESULTS:', n_correct, 'correct matches using top k:', top_k, ', percentage:', (100 * n_correct / n_cases), '%'

        top_k_mat.append(top_k_list)


    # Average the results
    n_cases_list_np = np.asarray(n_cases_list)
    total_cases = float(sum(n_cases_list_np))
    top_k_mat_np = np.asarray(top_k_mat)

    for i in range(n_experiments):
        top_k_mat_np[i, :] = top_k_mat_np[i, :] * (n_cases_list_np[i] / total_cases)

    averaged_top_k = top_k_mat_np[0, :]
    for i in range(1, n_experiments):
        averaged_top_k = averaged_top_k + top_k_mat_np[i, :]

    results_list = averaged_top_k.tolist()
    return results_list




# SERIES OF VALUES FOR TOP K

max_top_k = 20

# top_k_values = range(1, 32, 2)
top_k_values = range(1, max_top_k, 1)


# ------  Without Semantic Segmentation  ------

# My Single Part Manifold (Part 1)
g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_embedding_prototxt_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part1.prototxt'  # Is correct
image_embedding_caffemodel_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part1_iter_100000.caffemodel'
feat_name_part1 = 'image_embedding_part1'

# My Single Part Manifold (Part 2)
g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part2.txt'  # Is correct
image_embedding_prototxt_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part2.prototxt'  # Is correct
image_embedding_caffemodel_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part2_iter_100000.caffemodel'
feat_name_part2 = 'image_embedding_part2'

# My Single Part Manifold (Part 3)
g_shape_embedding_space_file_txt_part3 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part3.txt'  # Is correct
image_embedding_prototxt_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part3.prototxt'  # Is correct
image_embedding_caffemodel_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part3_iter_100000.caffemodel'
feat_name_part3 = 'image_embedding_part3'

# My Single Part Manifold (Part 4)
g_shape_embedding_space_file_txt_part4 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part4.txt'  # Is correct
image_embedding_prototxt_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part4.prototxt'  # Is correct
image_embedding_caffemodel_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part4_iter_100000.caffemodel'
feat_name_part4 = 'image_embedding_part4'


# ------  With Semantic Segmentation  ------

# My Single Part Manifold (Part 1)
g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_semSeg_embedding_prototxt_part1 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part1.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part1 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part1_iter_400000.caffemodel'
feat_name_part1 = 'image_embedding_part1'

# My Single Part Manifold (Part 2)
g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part2.txt'  # Is correct
image_semSeg_embedding_prototxt_part2 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part2.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part2 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part2_iter_400000.caffemodel'
feat_name_part2 = 'image_embedding_part2'

# My Single Part Manifold (Part 3)
g_shape_embedding_space_file_txt_part3 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part3.txt'  # Is correct
image_semSeg_embedding_prototxt_part3 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part3.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part3 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part3_iter_400000.caffemodel'
feat_name_part3 = 'image_embedding_part3'

# My Single Part Manifold (Part 4)
g_shape_embedding_space_file_txt_part4 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part4.txt'  # Is correct
image_semSeg_embedding_prototxt_part4 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part4.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part4 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part4_iter_400000.caffemodel'
feat_name_part4 = 'image_embedding_part4'


# Compute the blended results from the 4 manifolds at the same time
blended_part_results = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
                                                 g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
                                                 g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
                                                 g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
                                                 top_k_values)

blended_semSeg_part_results = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_semSeg_embedding_prototxt_part1, image_semSeg_embedding_caffemodel_part1, feat_name_part1,
                                                        g_shape_embedding_space_file_txt_part2, image_semSeg_embedding_prototxt_part2, image_semSeg_embedding_caffemodel_part2, feat_name_part2,
                                                        g_shape_embedding_space_file_txt_part3, image_semSeg_embedding_prototxt_part3, image_semSeg_embedding_caffemodel_part3, feat_name_part3,
                                                        g_shape_embedding_space_file_txt_part4, image_semSeg_embedding_prototxt_part4, image_semSeg_embedding_caffemodel_part4, feat_name_part4,
                                                        top_k_values)



font = {'family': 'normal', 'weight': 'bold', 'size': 20}
line_size = 3

plt.xlabel('Top-k', fontdict=font)
plt.ylabel('Accuracy', fontdict=font)
plt.title('ExactPartMatch Dataset results', fontdict=font)
plt.plot(top_k_values, blended_part_results, '--', color='#9acd32', linewidth=line_size, label='Li SiggAsia 2015 Parts, (Blended Parts)')
plt.plot(top_k_values, blended_semSeg_part_results, color='#20b2aa', linewidth=line_size, label='Ours, (Blended Parts)')
plt.legend(loc=4)

ax = plt.gca()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Customize the major grid
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth='0.4 ', color='red')  # Customize the minor grid
plt.axis([1, max_top_k-1, 0, 100])
plt.show()

lolo = 1


