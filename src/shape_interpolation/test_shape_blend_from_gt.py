#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from caffe.proto import caffe_pb2

from global_variables import *
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt


# Manifold part 1
g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part1.txt'  # Is correct
shape_embedding_space_part1 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt_part1, 'r')]
shape_embedding_space_part1_np = np.asarray(shape_embedding_space_part1)

# Manifold part 2
g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part2.txt'  # Is correct
shape_embedding_space_part2 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt_part2, 'r')]
shape_embedding_space_part2_np = np.asarray(shape_embedding_space_part2)

# Manifold part 3
g_shape_embedding_space_file_txt_part3 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part3.txt'  # Is correct
shape_embedding_space_part3 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt_part3, 'r')]
shape_embedding_space_part3_np = np.asarray(shape_embedding_space_part3)

# Manifold part 4
g_shape_embedding_space_file_txt_part4 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part4.txt'  # Is correct
shape_embedding_space_part4 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt_part4, 'r')]
shape_embedding_space_part4_np = np.asarray(shape_embedding_space_part4)


# ----------------          indexes to test the blending          ----------------
# part ID: 1 armrest, 2 back, 3 legs, 4 seat
n_shapes = shape_embedding_space_part2_np.shape[0]

# # everything except legs from 1a6f615e8b1b5ae4dbbc9440457e303e
# # legs from 1be0108997e6aba5349bb1cbbf9a4206
# md5_a1 = 183  # part 1, 2, 4
# md5_b1 = 219  # part 3
#
# # everything except legs from 1a74a83fa6d24b3cacd67ce2c72c02e
# # legs from 1be0108997e6aba5349bb1cbbf9a4206
# md5_a2 = 184  # part 1, 2, 4
# md5_b2 = 219  # part 3
#
# # back from 1b7ba5484399d36bc5e50b867ca2d0b9
# # legs from 1be0108997e6aba5349bb1cbbf9a4206
# md5_a3 = 203  # part 2
# md5_b3 = 219  # part 3
#
# # everything except legs from 1a74a83fa6d24b3cacd67ce2c72c02e
# # legs from 1e2e68813f004d8ff8b8d4a282992be4
# md5_a4 = 184  # part 1, 2, 4
# md5_b4 = 271  # part 3
#
# coord_part2 = shape_embedding_space_part2_np[md5_a1]
# coord_part3 = shape_embedding_space_part3_np[md5_b1]
#
#
# accumulator = np.zeros(n_shapes)
#
# dist_list_part2 = []
# for sample in shape_embedding_space_part2_np:
#     dist = np.linalg.norm(coord_part2-sample)
#     dist_list_part2.append(dist)
#
# sorted_shapes2 = np.argsort(dist_list_part2)
#
# for i in range(n_shapes):
#     accumulator[sorted_shapes2[i]] = i+1
#
#
# dist_list_part3 = []
# for sample in shape_embedding_space_part3_np:
#     dist = np.linalg.norm(coord_part3-sample)
#     dist_list_part3.append(dist)
#
# sorted_shapes3 = np.argsort(dist_list_part3)
#
# for i in range(n_shapes):
#     accumulator[sorted_shapes3[i]] = accumulator[sorted_shapes3[i]] + (i+1)
#
#
# best_candidates = np.argsort(accumulator)
#
#
# dist_list_part2_np = np.asarray(dist_list_part2)
# dist_list_part3_np = np.asarray(dist_list_part3)
# dist_list_np = dist_list_part2_np + dist_list_part3_np
#
# best_comb_dist = np.sort(dist_list_np)
# best_comb_indexes = np.argsort(dist_list_np)




# part ID: 1 armrest, 2 back, 3 legs, 4 seat

# armrests from 1a6f615e8b1b5ae4dbbc9440457e303e
# legs from 1e2e68813f004d8ff8b8d4a282992be4
md5_a5 = 183  # part 1
md5_b5 = 271  # part 3

coord_part1 = shape_embedding_space_part1_np[md5_a5]
coord_part3 = shape_embedding_space_part3_np[md5_b5]


accumulator = np.zeros(n_shapes)

dist_list_part1 = []
for sample in shape_embedding_space_part1_np:
    dist = np.linalg.norm(coord_part1-sample)
    dist_list_part1.append(dist)

sorted_shapes2 = np.argsort(dist_list_part1)

for i in range(n_shapes):
    accumulator[sorted_shapes2[i]] = i+1


dist_list_part3 = []
for sample in shape_embedding_space_part3_np:
    dist = np.linalg.norm(coord_part3-sample)
    dist_list_part3.append(dist)

sorted_shapes3 = np.argsort(dist_list_part3)

for i in range(n_shapes):
    accumulator[sorted_shapes3[i]] = accumulator[sorted_shapes3[i]] + (i+1)


best_candidates = np.argsort(accumulator)


dist_list_part1_np = np.asarray(dist_list_part1)
dist_list_part3_np = np.asarray(dist_list_part3)
dist_list_np = dist_list_part1_np + dist_list_part3_np

best_comb_dist = np.sort(dist_list_np)
best_comb_indexes = np.argsort(dist_list_np)



lolo = 1

