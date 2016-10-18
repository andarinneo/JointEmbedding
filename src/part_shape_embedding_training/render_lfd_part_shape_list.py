#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
import sys
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

# from utilities_math import *
# import scipy.io


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

report_step = 100

if __name__ == '__main__':
    if not os.path.exists(g_lfd_images_folder):
        os.mkdir(g_lfd_images_folder) 

    shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
    print(len(shape_list), ' shapes are going to be rendered!')

    #print('Generating rendering commands...', end = '')
    commands = []
    for shape_property in shape_list:
        shape_synset = shape_property[0]
        shape_md5 = shape_property[1]
        shape_file = os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'colored_parts.obj')

        command = '%s ../color_rendering.blend --background --python render_lfd_part_single_shape.py -- %s %s %s ' % (g_blender_executable_path, shape_file, shape_synset, shape_md5)
        if len(shape_list) > 32:
            command = command + ' > /dev/null 2>&1'
        commands.append(command)

        # break

    print('done(%d commands)'%(len(commands)))


    # # Save the poses of the rendering for each class
    # lfd_root_folder = os.path.join(g_lfd_images_folder, shape_synset)
    # for azimuth_deg in g_lfd_camera_azimuth_dict[shape_synset]:
    #     for elevation_deg in g_lfd_camera_elevation_dict[shape_synset]:
    #         theta_deg = 0
    #         lfd_pose_file = '%s_a%03d_e%03d_t%03d_d%03d.mat' % (shape_synset, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(g_lfd_camera_dist))
    #
    #         cx, cy, cz = obj_centened_camera_pos(g_lfd_camera_dist, azimuth_deg, elevation_deg)
    #         q1 = camPosToQuaternion(cx, cy, cz)
    #         q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    #         q = quaternionProduct(q2, q1)
    #
    #         scipy.io.savemat(os.path.join(lfd_root_folder, lfd_pose_file), mdict={'cx':cx, 'cy':cy, 'cz':cz, 'q0':q[0], 'q1':q[1], 'q2':q[2], 'q3':q[3]} )



    print('Rendering, it takes long time...')
    pool = Pool(g_lfd_rendering_thread_num)
    for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
        if idx % report_step == 0:
            print('[%s] Rendering command %d of %d' % (datetime.datetime.now().time(), idx, len(shape_list)))
        if return_code != 0:
            print('Rendering command %d of %d (\"%s\") failed' % (idx, len(shape_list), commands[idx]))
