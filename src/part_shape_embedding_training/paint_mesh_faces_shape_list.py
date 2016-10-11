#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from paint_mesh_faces_single_shape import *
from global_variables import *
# Parallel FOR library
from joblib import Parallel, delayed
import multiprocessing


def loop_operation(root_folder, shape_property_v, n, labels):
    class_id = shape_property_v[0]
    model_id = shape_property_v[1]

    paint_mesh_faces(root_folder, class_id, model_id, n, labels)


# --- MAIN ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

report_step = 100

n_parts = 4
part_labels = {0: 'arm', 1: 'back', 2: 'leg', 3: 'seat'}

if __name__ == '__main__':
    if not os.path.exists(g_lfd_images_folder):
        os.mkdir(g_lfd_images_folder)

    shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
    print(len(shape_list), ' shapes are going to be colored!')

    non_available_segmentations = []
    for shape_property in shape_list:
        shape_synset = shape_property[0]
        shape_md5 = shape_property[1]

        try:
            paint_mesh_faces(g_shapenet_root_folder, shape_synset, shape_md5, n_parts, part_labels)
        except:
            non_available_segmentations.append(shape_md5)
            print('Not able to produce labels for: ' + shape_md5)
        #break # BORRAR, solo para debugar

    #num_cores = multiprocessing.cpu_count()
    #Parallel(n_jobs=num_cores)(delayed(loop_operation)(g_shapenet_root_folder, shape_property, n_parts, part_labels) for shape_property in shape_list)


    print('done(%d models)'%(len(shape_list)))
