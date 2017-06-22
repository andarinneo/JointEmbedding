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


def run_top_k_blended_results(shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feature_name_part1,
                              shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feature_name_part2,
                              shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feature_name_part3,
                              shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feature_name_part4,
                              top_k_values, criteria):

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

    shape_list_np = np.asarray(shape_list)


    dist_mat = []
    for part in range(g_n_parts):
        shape_embedding_array = shape_embedding_space_np[part, shape_list_np, :]

        # ----------------          Evaluate image in network          ----------------
        image_embedding_array = extract_cnn_features(img_filelist=new_imgname_filelist_file,
                                                     img_root=path,
                                                     prototxt=image_embedding_prototxt[part],
                                                     caffemodel=image_embedding_caffemodel[part],
                                                     feat_name=feature_name[part],
                                                     caffe_path=g_caffe_install_path,
                                                     mean_file=g_mean_file)
        image_embedding_array = np.asarray(image_embedding_array)

        # ----------------          Compute distances between shape and estimation          ----------------
        dist_mat.append(distance_matrix(image_embedding_array, shape_embedding_array))


    # ----------------          Find distance to 105 3D model in manifold space          ----------------

    blended_dist_mat = dist_mat[0] + dist_mat[1] + dist_mat[2] + dist_mat[3]

    results_part1_manifold = np.asarray([dist_mat[0][i, :].argsort() for i in range(n_imgs)])
    results_part2_manifold = np.asarray([dist_mat[1][i, :].argsort() for i in range(n_imgs)])
    results_part3_manifold = np.asarray([dist_mat[2][i, :].argsort() for i in range(n_imgs)])
    results_part4_manifold = np.asarray([dist_mat[3][i, :].argsort() for i in range(n_imgs)])

    ranking_mat_part1 = np.zeros([n_imgs, n_shapes])
    ranking_mat_part2 = np.zeros([n_imgs, n_shapes])
    ranking_mat_part3 = np.zeros([n_imgs, n_shapes])
    ranking_mat_part4 = np.zeros([n_imgs, n_shapes])
    for img_idx in range(n_imgs):
        for shape_idx in range(n_shapes):
            ranking_idx = results_part1_manifold[img_idx, shape_idx]
            ranking_mat_part1[img_idx, ranking_idx] = shape_idx
            ranking_idx = results_part2_manifold[img_idx, shape_idx]
            ranking_mat_part2[img_idx, ranking_idx] = shape_idx
            ranking_idx = results_part3_manifold[img_idx, shape_idx]
            ranking_mat_part3[img_idx, ranking_idx] = shape_idx
            ranking_idx = results_part4_manifold[img_idx, shape_idx]
            ranking_mat_part4[img_idx, ranking_idx] = shape_idx

    # blended_ranking_mat = ranking_mat_part1 + ranking_mat_part2 + ranking_mat_part3 + ranking_mat_part4

    if criteria == 1:
        blended_mat = blended_dist_mat
    elif criteria == 2:
        blended_dist_mat = dist_mat[1] + dist_mat[2]
        blended_mat = blended_dist_mat
    elif criteria == 3:
        blended_dist_mat = dist_mat[0] + dist_mat[1] + dist_mat[2]
        blended_mat = blended_dist_mat
    elif criteria == 4:
        blended_ranking_mat = ranking_mat_part2 + ranking_mat_part3 + ranking_mat_part4
        blended_mat = blended_ranking_mat
    elif criteria == 5:
        blended_ranking_mat = ranking_mat_part2 + ranking_mat_part3
        blended_mat = blended_ranking_mat


    results_list = []
    for top_k in top_k_values:
        results_blended = np.asarray([blended_mat[i, :].argsort()[:top_k] for i in range(n_imgs)])

        bool_vec = []
        for i in range(n_imgs):
            aux_vec = shape_list_np[results_blended[:][i]]
            bool_val = bool(aux_vec[np.where(aux_vec == gt_img2shape[i])].shape[0])
            bool_vec.append(bool_val)

        n_correct = sum(bool_vec)
        results_list.append((100 * n_correct / n_imgs))
        print 'RESULTS:', n_correct, 'correct matches using top k:', top_k, ', percentage:', (100 * n_correct / n_imgs), '%'

    return results_list



# SERIES OF VALUES FOR TOP K
top_k_values = range(1, 32, 2)


# GT Single Manifold (including test shapes)
g_shape_embedding_space_file_txt = '/home/adrian/Desktop/03001627/shape_embedding_space_03001627.txt'  # Is correct
image_embedding_prototxt = '/home/adrian/Desktop/03001627/image_embedding_03001627.prototxt'  # Is correct
image_embedding_caffemodel = '/home/adrian/Desktop/03001627/image_embedding_03001627.caffemodel'
feat_name = 'image_embedding'

# gt_results_its = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Single Manifold (including test shapes)
g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/shape_embedding_space_03001627(myTraining).txt'  # Is correct
image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/image_embedding_rcnn.prototxt'  # Is correct
image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_20000(itWorks).caffemodel'
feat_name = 'image_embedding'

# single_results_its = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Single Manifold
g_shape_embedding_space_file_txt = '/home/adrian/Desktop/03001627/shape_embedding_space_03001627.txt'  # Is correct
image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/image_embedding_rcnn.prototxt'  # Is correct
image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_40000.caffemodel'
feat_name = 'image_embedding'

# single_results = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Single Part Manifold (part 1, including test shapes)
g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/backup/combined_part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part1.prototxt'  # Is correct
image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_100000_its.caffemodel'
feat_name = 'image_embedding_part1'

# single_part_results_its = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Single Part Manifold (Part 1)
g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_embedding_prototxt_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part1.prototxt'  # Is correct
image_embedding_caffemodel_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part1_iter_100000.caffemodel'
feat_name_part1 = 'image_embedding_part1'

# single_part1_results = run_top_k_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1, top_k_values)


# My Single Part Manifold (Part 2)
g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part2.txt'  # Is correct
image_embedding_prototxt_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part2.prototxt'  # Is correct
image_embedding_caffemodel_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part2_iter_100000.caffemodel'
feat_name_part2 = 'image_embedding_part2'

# single_part2_results = run_top_k_results(g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2, top_k_values)


# My Single Part Manifold (Part 3)
g_shape_embedding_space_file_txt_part3 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part3.txt'  # Is correct
image_embedding_prototxt_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part3.prototxt'  # Is correct
image_embedding_caffemodel_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part3_iter_100000.caffemodel'
feat_name_part3 = 'image_embedding_part3'

# single_part3_results = run_top_k_results(g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3, top_k_values)


# My Single Part Manifold (Part 4)
g_shape_embedding_space_file_txt_part4 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part4.txt'  # Is correct
image_embedding_prototxt_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part4.prototxt'  # Is correct
image_embedding_caffemodel_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part4_iter_100000.caffemodel'
feat_name_part4 = 'image_embedding_part4'

# single_part4_results = run_top_k_results(g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4, top_k_values)


# # Compute the blended results from the 4 manifolds at the same time
# blended_part_results_c1 = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
#                                                  g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
#                                                  g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
#                                                  g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
#                                                  top_k_values, 1)

blended_part_results_c2 = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
                                                 g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
                                                 g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
                                                 g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
                                                 top_k_values, 2)

# blended_part_results_c3 = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
#                                                  g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
#                                                  g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
#                                                  g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
#                                                  top_k_values, 3)
#
# blended_part_results_c4 = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
#                                                  g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
#                                                  g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
#                                                  g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
#                                                  top_k_values, 4)
#
# blended_part_results_c5 = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
#                                                  g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
#                                                  g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
#                                                  g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
#                                                  top_k_values, 5)



plt.xlabel('top k retrieval')
plt.ylabel('Probability')
plt.title('ExactMatch Dataset results')
plt.plot(top_k_values, gt_results_its, 'k', label='GT Whole Chair (Includes test shapes)')
plt.plot(top_k_values, single_results_its, 'b', label='My Whole Chair (Includes test shapes)')
plt.plot(top_k_values, single_results, 'g', label='My Whole Chair')
plt.plot(top_k_values, single_part_results_its, 'y', label='Single Part 1="armrest" (Includes test shapes)')
plt.plot(top_k_values, single_part1_results, 'g--', label='Single Part 1="armrest"')
plt.plot(top_k_values, single_part2_results, 'r--', label='Single Part 2="back"')
plt.plot(top_k_values, single_part3_results, 'b--', label='Single Part 3="legs"')
plt.plot(top_k_values, single_part4_results, 'k--', label='Single Part 4="seat"')
# plt.plot(top_k_values, blended_part_results_c1, 'r-', label='Blended Parts, (Blended dist matrices 1.2.3.4)')
plt.plot(top_k_values, blended_part_results_c2, 'r+-', label='Blended Parts, (Blended dist matrices 2.3)')
# plt.plot(top_k_values, blended_part_results_c3, 'r*-', label='Blended Parts, (Blended dist matrices 1.2.3)')
# plt.plot(top_k_values, blended_part_results_c4, 'r8-', label='Blended Parts, (blended rankings 2.3.4)')
# plt.plot(top_k_values, blended_part_results_c5, 'r>-', label='Blended Parts, (blended rankings 2.3)')
plt.legend(loc=4)

ax = plt.gca()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Customize the major grid
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth='0.4 ', color='red')  # Customize the minor grid
plt.axis([0, 32, 0, 100])
plt.show()

lolo = 1


