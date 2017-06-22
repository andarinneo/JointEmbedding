#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_caffe import extract_cnn_features


extract_cnn_features(img_filelist=g_syn_images_filelist,
                     img_root='/',
                     prototxt=g_extract_feat_manifold_prototxt,
                     caffemodel=g_fine_tune_manifold_caffemodel,
                     feat_name='concat_rgb_heat_maps',
                     output_path=g_concat_rgb_heat_maps_lmdb,
                     output_type='lmdb',
                     caffe_path=g_caffe_install_path,
                     mean_file=g_mean_file,
                     gpu_index=g_extract_feat_gpu_index,
                     pool_size=g_extract_feat_thread_num)

