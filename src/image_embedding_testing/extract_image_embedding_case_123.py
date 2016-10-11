#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import argparse
from google.protobuf import text_format

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_caffe import *

parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
parser.add_argument('--image', help='Path to input image (cropped)', required=False, default='/home/adrian/JointEmbedding/src/image_embedding_testing/testing_images/chair_00.jpg')
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False, default='/home/adrian/Desktop/03001627/image_embedding_03001627.caffemodel')
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False, default='/home/adrian/Desktop/03001627/image_embedding_03001627.prototxt')
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
args = parser.parse_args()

image_embedding_caffemodel = os.path.join(g_image_embedding_testing_folder, 'snapshots%s_iter_%d.caffemodel'%(g_shapenet_synset_set_handle, args.iter_num))
image_embedding_prototxt = g_image_embedding_testing_prototxt

if args.caffemodel:
    image_embedding_caffemodel = args.caffemodel
if args.prototxt:
    image_embedding_prototxt = args.prototxt



#g_caffe_install_path='/home/adrian/caffe/build/tools/caffe/'

image0 = '/home/adrian/JointEmbedding/src/image_embedding_testing/testing_images/chair_00.jpg'
image1 = '/home/adrian/JointEmbedding/src/image_embedding_testing/testing_images/chair_01.jpg'
image2 = '/home/adrian/JointEmbedding/src/image_embedding_testing/testing_images/chair_02.jpg'
image3 = '/home/adrian/JointEmbedding/src/image_embedding_testing/testing_images/chair_03.jpg'
image4 = '/home/adrian/JointEmbedding/src/image_embedding_testing/testing_images/chair_04.jpg'


print 'Image embedding for %s is:'%(image0)
image_embedding_array = extract_cnn_features(img_filelist=image0,
                                             img_root='/',
                                             prototxt=image_embedding_prototxt,
                                             caffemodel=image_embedding_caffemodel,
                                             feat_name='image_embedding',
                                             caffe_path=g_caffe_install_path,
                                             mean_file=g_mean_file)[0]

coordinates0 = np.asarray(image_embedding_array.tolist())


print 'Image embedding for %s is:'%(image1)
image_embedding_array = extract_cnn_features(img_filelist=image1,
                                             img_root='/',
                                             prototxt=image_embedding_prototxt,
                                             caffemodel=image_embedding_caffemodel,
                                             feat_name='image_embedding',
                                             caffe_path=g_caffe_install_path,
                                             mean_file=g_mean_file)[0]

coordinates1 = np.asarray(image_embedding_array.tolist())


print 'Image embedding for %s is:'%(image2)
image_embedding_array = extract_cnn_features(img_filelist=image2,
                                             img_root='/',
                                             prototxt=image_embedding_prototxt,
                                             caffemodel=image_embedding_caffemodel,
                                             feat_name='image_embedding',
                                             caffe_path=g_caffe_install_path,
                                             mean_file=g_mean_file)[0]

coordinates2 = np.asarray(image_embedding_array.tolist())


print 'Image embedding for %s is:'%(image3)
image_embedding_array = extract_cnn_features(img_filelist=image3,
                                             img_root='/',
                                             prototxt=image_embedding_prototxt,
                                             caffemodel=image_embedding_caffemodel,
                                             feat_name='image_embedding',
                                             caffe_path=g_caffe_install_path,
                                             mean_file=g_mean_file)[0]

coordinates3 = np.asarray(image_embedding_array.tolist())


print 'Image embedding for %s is:'%(image4)
image_embedding_array = extract_cnn_features(img_filelist=image4,
                                             img_root='/',
                                             prototxt=image_embedding_prototxt,
                                             caffemodel=image_embedding_caffemodel,
                                             feat_name='image_embedding',
                                             caffe_path=g_caffe_install_path,
                                             mean_file=g_mean_file)[0]

coordinates4 = np.asarray(image_embedding_array.tolist())


dist21 = np.linalg.norm(coordinates2 - coordinates1)
dist23 = np.linalg.norm(coordinates2 - coordinates3) # should be min
dist31 = np.linalg.norm(coordinates3 - coordinates1)


dist04 = np.linalg.norm(coordinates4 - coordinates0) # should be min
dist01 = np.linalg.norm(coordinates3 - coordinates0)
dist02 = np.linalg.norm(coordinates2 - coordinates0)
dist03 = np.linalg.norm(coordinates1 - coordinates0)


                     
print image_embedding_array.tolist()
