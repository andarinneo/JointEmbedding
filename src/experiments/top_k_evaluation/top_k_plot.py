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
        blended_dist_mat = dist_mat[1] + dist_mat[2]
        blended_mat = blended_dist_mat
    elif criteria == 2:
        blended_dist_mat = dist_mat[0] + dist_mat[1] + dist_mat[2]
        blended_mat = blended_dist_mat
    elif criteria == 3:
        blended_dist_mat = dist_mat[0] + dist_mat[1] + dist_mat[2] + dist_mat[3]
        blended_mat = blended_dist_mat
    elif criteria == 4:
        blended_dist_mat = dist_mat[0] + dist_mat[2]
        blended_mat = blended_dist_mat


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

single_results = run_top_k_results(g_shape_embedding_space_file_txt, image_embedding_prototxt, image_embedding_caffemodel, feat_name, top_k_values)


# My Single Blended Part Manifold (Part 1)
g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/backup/combined_part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_semSeg_embedding_prototxt_part1 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part1.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part1 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part1_iter_400000.caffemodel'
feat_name_part1 = 'image_embedding_part1'

single_part1_results_its = run_top_k_results(g_shape_embedding_space_file_txt_part1, image_semSeg_embedding_prototxt_part1, image_semSeg_embedding_caffemodel_part1, feat_name_part1, top_k_values)


# My Single Blended Part Manifold (part 2, including test shapes)
g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/backup/combined_part_shape_embedding_space_03001627_part2.txt'  # Is correct
image_semSeg_embedding_prototxt_part2 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part2.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part2 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part2_iter_400000.caffemodel'
feat_name_part2 = 'image_embedding_part2'

single_part2_results_its = run_top_k_results(g_shape_embedding_space_file_txt_part2, image_semSeg_embedding_prototxt_part2, image_semSeg_embedding_caffemodel_part2, feat_name_part2, top_k_values)


# My Single Blended Part Manifold (part 3, including test shapes)
g_shape_embedding_space_file_txt_part3 = '/media/adrian/Datasets/datasets/shape_embedding/backup/combined_part_shape_embedding_space_03001627_part3.txt'  # Is correct
image_semSeg_embedding_prototxt_part3 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part3.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part3 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part3_iter_400000.caffemodel'
feat_name_part3 = 'image_embedding_part3'

single_part3_results_its = run_top_k_results(g_shape_embedding_space_file_txt_part3, image_semSeg_embedding_prototxt_part3, image_semSeg_embedding_caffemodel_part3, feat_name_part3, top_k_values)


# My Single Blended Part Manifold (part 4, including test shapes)
g_shape_embedding_space_file_txt_part4 = '/media/adrian/Datasets/datasets/shape_embedding/backup/combined_part_shape_embedding_space_03001627_part4.txt'  # Is correct
image_semSeg_embedding_prototxt_part4 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part4.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part4 = '/home/adrian/JointEmbedding/datasets/image_embedding/combinedShape_part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part4_iter_400000.caffemodel'
feat_name_part4 = 'image_embedding_part4'

single_part4_results_its = run_top_k_results(g_shape_embedding_space_file_txt_part4, image_semSeg_embedding_prototxt_part4, image_semSeg_embedding_caffemodel_part4, feat_name_part4, top_k_values)

criteria = 4
blended_whole_and_part_results = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_semSeg_embedding_prototxt_part1, image_semSeg_embedding_caffemodel_part1, feat_name_part1,
                                                           g_shape_embedding_space_file_txt_part2, image_semSeg_embedding_prototxt_part2, image_semSeg_embedding_caffemodel_part2, feat_name_part2,
                                                           g_shape_embedding_space_file_txt_part3, image_semSeg_embedding_prototxt_part3, image_semSeg_embedding_caffemodel_part3, feat_name_part3,
                                                           g_shape_embedding_space_file_txt_part4, image_semSeg_embedding_prototxt_part4, image_semSeg_embedding_caffemodel_part4, feat_name_part4,
                                                           top_k_values, criteria)



# ------  Without Semantic Segmentation  ------

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


# ------  With Semantic Segmentation  ------

# My Single Part Manifold (Part 1)
g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part1.txt'  # Is correct
image_semSeg_embedding_prototxt_part1 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part1.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part1 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part1_iter_400000.caffemodel'
feat_name_part1 = 'image_embedding_part1'

# single_semSeg_part1_results = run_top_k_results(g_shape_embedding_space_file_txt_part1, image_semSeg_embedding_prototxt_part1, image_semSeg_embedding_caffemodel_part1, feat_name_part1, top_k_values)


# My Single Part Manifold (Part 2)
g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part2.txt'  # Is correct
image_semSeg_embedding_prototxt_part2 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part2.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part2 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part2_iter_400000.caffemodel'
feat_name_part2 = 'image_embedding_part2'

# single_semSeg_part2_results = run_top_k_results(g_shape_embedding_space_file_txt_part2, image_semSeg_embedding_prototxt_part2, image_semSeg_embedding_caffemodel_part2, feat_name_part2, top_k_values)


# My Single Part Manifold (Part 3)
g_shape_embedding_space_file_txt_part3 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part3.txt'  # Is correct
image_semSeg_embedding_prototxt_part3 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part3.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part3 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part3_iter_400000.caffemodel'
feat_name_part3 = 'image_embedding_part3'

# single_semSeg_part3_results = run_top_k_results(g_shape_embedding_space_file_txt_part3, image_semSeg_embedding_prototxt_part3, image_semSeg_embedding_caffemodel_part3, feat_name_part3, top_k_values)


# My Single Part Manifold (Part 4)
g_shape_embedding_space_file_txt_part4 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part4.txt'  # Is correct
image_semSeg_embedding_prototxt_part4 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part4.prototxt'  # Is correct
image_semSeg_embedding_caffemodel_part4 = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part4_iter_400000.caffemodel'
feat_name_part4 = 'image_embedding_part4'

# single_semSeg_part4_results = run_top_k_results(g_shape_embedding_space_file_txt_part4, image_semSeg_embedding_prototxt_part4, image_semSeg_embedding_caffemodel_part4, feat_name_part4, top_k_values)




# # Compute the blended results from the 4 manifolds at the same time
criteria = 2  # all parts except the seat

blended_part_results = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_embedding_prototxt_part1, image_embedding_caffemodel_part1, feat_name_part1,
                                                 g_shape_embedding_space_file_txt_part2, image_embedding_prototxt_part2, image_embedding_caffemodel_part2, feat_name_part2,
                                                 g_shape_embedding_space_file_txt_part3, image_embedding_prototxt_part3, image_embedding_caffemodel_part3, feat_name_part3,
                                                 g_shape_embedding_space_file_txt_part4, image_embedding_prototxt_part4, image_embedding_caffemodel_part4, feat_name_part4,
                                                 top_k_values, criteria)

blended_semSeg_part_results = run_top_k_blended_results(g_shape_embedding_space_file_txt_part1, image_semSeg_embedding_prototxt_part1, image_semSeg_embedding_caffemodel_part1, feat_name_part1,
                                                        g_shape_embedding_space_file_txt_part2, image_semSeg_embedding_prototxt_part2, image_semSeg_embedding_caffemodel_part2, feat_name_part2,
                                                        g_shape_embedding_space_file_txt_part3, image_semSeg_embedding_prototxt_part3, image_semSeg_embedding_caffemodel_part3, feat_name_part3,
                                                        g_shape_embedding_space_file_txt_part4, image_semSeg_embedding_prototxt_part4, image_semSeg_embedding_caffemodel_part4, feat_name_part4,
                                                        top_k_values, criteria)



font = {'family': 'normal', 'weight': 'bold', 'size': 20}
line_size = 3

plt.xlabel('Top-k', fontdict=font)
plt.ylabel('Accuracy', fontdict=font)
plt.title('ExactMatch Dataset results', fontdict=font)
# plt.plot(top_k_values, gt_results_its, 'g', label='GT Whole Chair (Includes test shapes)')
# plt.plot(top_k_values, single_results_its, 'b', label='My Whole Chair (Includes test shapes)')
plt.plot(top_k_values, single_results, '--', color='#0000ff', linewidth=line_size, label='Li, SiggAsia 2015')
plt.plot(top_k_values, [55, 70, 75, 80, 82, 82, 83, 85, 86, 86, 87, 91, 91, 93, 94, 94], '--', color='#ff0000', linewidth=line_size, label='HoG')
plt.plot(top_k_values, [38, 55, 68, 75, 80, 82, 85, 86, 89, 90, 92, 94, 94, 95, 95, 95], '--', color='#800080', linewidth=line_size, label='AlexNet (fine tune)')
plt.plot(top_k_values, [33, 53, 66, 70, 75, 76, 77, 80, 82, 84, 85, 86, 86, 86, 87, 87], '--', color='#ffa500', linewidth=line_size, label='Siamese (0 nbor)')
plt.plot(top_k_values, [27, 46, 52, 55, 60, 61, 63, 64, 66, 67, 72, 75, 78, 81, 82, 83], '--', color='#00bfbf', linewidth=line_size, label='Siamese (64 nbor)')
plt.plot([10], [82], 'o', color='#000000', markersize=10, label='Girdhar, ECCV 2016')  # This is the only result provided in the supplemental material
plt.plot(top_k_values, blended_part_results, '--', color='#9acd32', linewidth=line_size, label='Li SiggAsia 2015 Parts, (Blended Parts)')
plt.plot(top_k_values, blended_semSeg_part_results, color='#20b2aa', linewidth=line_size, label='Ours, (Blended Parts)')
# plt.plot(top_k_values, single_part1_results_its, '--', color='#dAf7A6', linewidth=line_size, label='Part1, (Blended Whole+Parts)')
# plt.plot(top_k_values, single_part2_results_its, '--', color='#ffC300', linewidth=line_size, label='Part2, (Blended Whole+Parts)')
# plt.plot(top_k_values, single_part3_results_its, '--', color='#ff5733', linewidth=line_size, label='Part3, (Blended Whole+Parts)')
# plt.plot(top_k_values, single_part4_results_its, '--', color='#900c3f', linewidth=line_size, label='Part4, (Blended Whole+Parts)')
plt.plot(top_k_values, blended_whole_and_part_results, color='#901266', linewidth=line_size, label='(Blended Whole+Parts)')
plt.legend(loc=4)

ax = plt.gca()
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')  # Customize the major grid
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth='0.4 ', color='red')  # Customize the minor grid
plt.axis([1, 31, 0, 100])
plt.show()

lolo = 1


