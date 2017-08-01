#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_caffe import extract_cnn_features, stack_caffe_models


# Contains the feature layer training data
# g_extract_feat_manifold_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_training_03001627_rcnn/train_val_rcnn.prototxt'
extract_feat_manifold_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_training_03001627_rcnn/snapshots/snapshots_03001627_iter_100.caffemodel'


stack_caffe_models(prototxt=g_extract_feat_pool5_prototxt,
                   base_model=g_fine_tune_manifold_caffemodel,
                   top_model=extract_feat_manifold_caffemodel,
                   stacked_model=g_extract_feat_pool5_caffemodel,
                   caffe_path=g_caffe_install_path)


extract_cnn_features(img_filelist=g_syn_images_filelist,
                     img_root='/',
                     prototxt=g_extract_feat_pool5_prototxt,
                     caffemodel=g_extract_feat_pool5_caffemodel,
                     feat_name='manifold_pool5',
                     output_path=g_pool5_semSeg_lmdb,
                     output_type='lmdb',
                     caffe_path=g_caffe_install_path,
                     mean_file=g_mean_file,
                     gpu_index=g_extract_feat_gpu_index,
                     pool_size=g_extract_feat_thread_num)

