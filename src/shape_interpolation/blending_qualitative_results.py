import os
import sys
import shutil
import argparse
import fileinput
import numpy as np

from global_variables import *
from utilities_caffe import *

from shape_interpolation.blending_utilities import blend_2_inputs

matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))


# --------------------------               Predefined Parameters (No semSeg)                ----------------------------
# image_embedding_caffemodel_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part1_iter_100000.caffemodel'
# image_embedding_prototxt_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part1.prototxt'
# image_embedding_caffemodel_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part2_iter_100000.caffemodel'
# image_embedding_prototxt_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part2.prototxt'
# image_embedding_caffemodel_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part3_iter_100000.caffemodel'
# image_embedding_prototxt_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part3.prototxt'
# image_embedding_caffemodel_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part4_iter_100000.caffemodel'
# image_embedding_prototxt_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part4.prototxt'


# --------------------------              Predefined Parameters (With semSeg)               ----------------------------
image_embedding_caffemodel_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part1_iter_400000.caffemodel'
image_embedding_prototxt_part1 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part1.prototxt'
image_embedding_caffemodel_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part2_iter_400000.caffemodel'
image_embedding_prototxt_part2 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part2.prototxt'
image_embedding_caffemodel_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part3_iter_400000.caffemodel'
image_embedding_prototxt_part3 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part3.prototxt'
image_embedding_caffemodel_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/stacked_03001627_part4_iter_400000.caffemodel'
image_embedding_prototxt_part4 = '/media/adrian/Datasets/datasets/image_embedding/part_image_semSeg_embedding_testing_03001627_manifoldNet/image_embedding_manifoldNet_part4.prototxt'




codes_4_parts = {1: 'Armrests', 2: 'Back', 3: 'Legs', 4: 'Seat'}
part1_id = 2
part2_id = 4

for case_id in range(1, 164+1):  # +1 because index goes from 1..(N-1)
    path = '/home/adrian/Desktop/qualitative_blending_results/part' + str(part1_id) + '+part' + str(part2_id) + '/case' + str(case_id) + '/'

    if part1_id == 1:
        caffemodel1 = image_embedding_caffemodel_part1
        prototxt1 = image_embedding_prototxt_part1
    elif part1_id == 2:
        caffemodel1 = image_embedding_caffemodel_part2
        prototxt1 = image_embedding_prototxt_part2
    elif part1_id == 3:
        caffemodel1 = image_embedding_caffemodel_part3
        prototxt1 = image_embedding_prototxt_part3
    elif part1_id == 4:
        caffemodel1 = image_embedding_caffemodel_part4
        prototxt1 = image_embedding_prototxt_part4

    if part2_id == 2:
        caffemodel2 = image_embedding_caffemodel_part2
        prototxt2 = image_embedding_prototxt_part2
    elif part2_id == 3:
        caffemodel2 = image_embedding_caffemodel_part3
        prototxt2 = image_embedding_prototxt_part3
    elif part2_id == 4:
        caffemodel2 = image_embedding_caffemodel_part4
        prototxt2 = image_embedding_prototxt_part4


    # ---------------------------------                Load Parameters                 ---------------------------------
    parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
    parser.add_argument('--image1', help='Path to input image (cropped)', required=False, default= path + 'part' + str(part1_id) + '.jpg')
    parser.add_argument('--part_id1', help='Part Id for img1', type=int, default=part1_id)
    parser.add_argument('--iter_num1', '-n1', help='Use caffemodel trained after iter_num iterations', type=int, default=100000)
    parser.add_argument('--caffemodel1', '-c1', help='Path to caffemodel (will ignore -n option if provided)', required=False, default=caffemodel1)
    parser.add_argument('--prototxt1', '-p1', help='Path to prototxt (if not at the default place)', required=False, default=prototxt1)

    parser.add_argument('--image2', help='Path to input image (cropped)', required=False, default= path + 'part' + str(part2_id) + '.jpg')
    parser.add_argument('--part_id2', help='Part Id for img2', type=int, default=part2_id)
    parser.add_argument('--iter_num2', '-n2', help='Use caffemodel trained after iter_num iterations', type=int, default=100000)
    parser.add_argument('--caffemodel2', '-c2', help='Path to caffemodel (will ignore -n option if provided)', required=False, default=caffemodel2)
    parser.add_argument('--prototxt2', '-p2', help='Path to prototxt (if not at the default place)', required=False, default=prototxt2)


    parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
    parser.add_argument('--top_k', help='Retrieve top K shapes.', type=int, default=31)
    args = parser.parse_args()

    if args.caffemodel1:
        image_embedding_caffemodel1 = args.caffemodel1
    if args.prototxt1:
        image_embedding_prototxt1 = args.prototxt1

    if args.caffemodel2:
        image_embedding_caffemodel2 = args.caffemodel2
    if args.prototxt1:
        image_embedding_prototxt2 = args.prototxt2

    g_shape_embedding_space_file_txt_part1 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part' + str(args.part_id1) + '.txt'  # Is correct
    g_shape_embedding_space_file_txt_part2 = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part' + str(args.part_id2) + '.txt'  # Is correct


    # ---------------------------------             Find image 1 embedding             ---------------------------------

    print 'Computing image 1 embedding for %s...' % args.image1
    image_embedding_array1 = extract_cnn_features(img_filelist=args.image1,
                                                  img_root='/',
                                                  prototxt=image_embedding_prototxt1,
                                                  caffemodel=image_embedding_caffemodel1,
                                                  feat_name='image_embedding_part' + str(args.part_id1),
                                                  caffe_path=g_caffe_install_path,
                                                  mean_file=g_mean_file)
    image_embedding1 = image_embedding_array1[0]

    print 'Loading shape embedding space from %s...' % (g_shape_embedding_space_file_txt_part1)
    shape_embedding_space1 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt_part1, 'r')]
    assert (image_embedding1.size == shape_embedding_space1[0].size)


    # ---------------------------------             Find image 2 embedding             ---------------------------------

    print 'Computing image 2 embedding for %s...' % args.image2
    image_embedding_array2 = extract_cnn_features(img_filelist=args.image2,
                                                  img_root='/',
                                                  prototxt=image_embedding_prototxt2,
                                                  caffemodel=image_embedding_caffemodel2,
                                                  feat_name='image_embedding_part' + str(args.part_id2),
                                                  caffe_path=g_caffe_install_path,
                                                  mean_file=g_mean_file)
    image_embedding2 = image_embedding_array2[0]

    print 'Loading shape embedding space from %s...' % (g_shape_embedding_space_file_txt_part2)
    shape_embedding_space2 = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt_part2, 'r')]
    assert (image_embedding2.size == shape_embedding_space2[0].size)

    sorted_indexes = blend_2_inputs(image_embedding1, np.asarray(shape_embedding_space1), image_embedding2, np.asarray(shape_embedding_space2))


    # ---------------------------------             Generate visualization             ---------------------------------

    print 'Loading shape list from %s' % (g_shape_list_file)
    shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
    assert (len(shape_list) == len(shape_embedding_space1))


    visualization_filename = path + 'image_based_' + codes_4_parts[args.part_id1] + '+' + codes_4_parts[args.part_id2] + '_blending_case' + str(case_id) + '.html'
    print 'Saving visualization to %s...' % (visualization_filename)
    visualization_template = os.path.join(BASE_DIR, 'image_based_part_blending.html')
    shutil.copy(visualization_template, visualization_filename)

    for line in fileinput.input(visualization_filename, inplace=True):
        line = line.replace('PART_ID1', codes_4_parts[args.part_id1])
        line = line.replace('QUERY_IMAGE_FILENAME1', os.path.split(args.image1)[-1])
        sys.stdout.write(line)

    for line in fileinput.input(visualization_filename, inplace=True):
        line = line.replace('PART_ID2', codes_4_parts[args.part_id2])
        line = line.replace('QUERY_IMAGE_FILENAME2', os.path.split(args.image2)[-1])
        sys.stdout.write(line)

    retrieval_list = ''
    for i in range(args.top_k):
        shape_idx = sorted_indexes[i]
        synset = shape_list[shape_idx][0]
        md5_id = shape_list[shape_idx][1]
        retrieval_list = retrieval_list + \
                         """
                              <div class="retrieval">
                                 <span class="helper"></span>
                                 <img class="item" src="https://shapenet.cs.stanford.edu/shapenet_brain/media/shape_lfd_images/%s/%s/%s_%s_a054_e020_t000_d003.png" title="%s/%s">
                                 <div class="property">
                                 <p>id: %s</p>
                                 </div>
                             </div>
                          """ % (synset, md5_id, synset, md5_id, synset, md5_id, md5_id)

    for line in fileinput.input(visualization_filename, inplace=True):
        line = line.replace('RETRIEVAL_LIST', retrieval_list)
        sys.stdout.write(line)
