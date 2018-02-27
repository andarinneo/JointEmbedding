#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

filelist = [line.strip() for line in open(g_syn_images_filelist, 'r')]
imageid2shapeid = [line.strip() for line in open(g_syn_images_imageid2shapeid, 'r')]
labels_filelist = [line.strip() for line in open(g_syn_labels_filelist, 'r')]

image_num = len(filelist)
train_val_split = [1]*image_num
val_num = int(image_num*(1-g_train_ratio))
train_val_split[0:val_num] = [0]*val_num

print 'Training images: ', image_num - val_num
print 'Validation images: ', val_num

random.seed(9527)  # seed random with a fixed number
random.shuffle(train_val_split)

filelist_train = open(g_syn_images_filelist_train, 'w')
filelist_val = open(g_syn_images_filelist_val, 'w')
imageid2shapeid_train = open(g_syn_images_imageid2shapeid_train, 'w')
imageid2shapeid_val = open(g_syn_images_imageid2shapeid_val, 'w')
labels_filelist_train = open(g_syn_labels_filelist_train, 'w')
labels_filelist_val = open(g_syn_labels_filelist_val, 'w')
shuffled_image_indexes = open(g_shuffled_image_indexes, 'w')
train_val_split_file = open(g_syn_images_train_val_split, 'w')


# We need to shuffle the data to avoid the training batches to have only a couple of 3D models
# -- could this break the invariance to viewpoint learning?
image_indexes = range(image_num)
random.shuffle(image_indexes)


# Remove the samples included in the ExactMatchChair dataset
# Load
# shape_names_file = '/home/adrian/JointEmbedding/src/experiments/ExactMatchChairsDataset/filelist_exactmatch_chair_105.txt'
# shape_names = [line.strip() for line in open(shape_names_file, 'r')]

# counter = 0
# for idx, train_val in enumerate(train_val_split):
for idx in image_indexes:
    # # ExactMatchChair dataset
    # filename = filelist[idx]
    # if any(x in filename for x in shape_names):
    #     counter += 1
    # else:

    # Not ExactMatchChair dataset, if you want to eliminate those samples indent the code again
    train_val = train_val_split[idx]
    if train_val:
        filelist_train.write(filelist[idx]+'\n')
        imageid2shapeid_train.write(imageid2shapeid[idx]+'\n')
        labels_filelist_train.write(labels_filelist[idx]+'\n')
    else:
        filelist_val.write(filelist[idx]+'\n')
        imageid2shapeid_val.write(imageid2shapeid[idx]+'\n')
        labels_filelist_val.write(labels_filelist[idx]+'\n')
    shuffled_image_indexes.write(str(idx)+'\n')
    train_val_split_file.write(str(train_val)+'\n')


filelist_train.close()
filelist_val.close()
imageid2shapeid_train.close()
imageid2shapeid_val.close()
labels_filelist_train.close()
labels_filelist_val.close()
shuffled_image_indexes.close()
train_val_split_file.close()
