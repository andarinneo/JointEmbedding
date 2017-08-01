import os
import sys
import shutil
import argparse
import fileinput
import numpy as np
import caffe
from google.protobuf import text_format

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_caffe import *

parser = argparse.ArgumentParser(description="Extract image embedding features for IMAGE input.")
parser.add_argument('--image', help='Path to input image (cropped)', required=False, default='/home/adrian/Desktop/qualitative_blending_results/ExactPartMatch_semSeg/fae27953e0f0404ca99df2794ec76201_2.jpg')
parser.add_argument('--iter_num', '-n', help='Use caffemodel trained after iter_num iterations', type=int, default=20000)
# '/home/adrian/Desktop/03001627/image_embedding_03001627.caffemodel'
# '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_40000.caffemodel'
parser.add_argument('--caffemodel', '-c', help='Path to caffemodel (will ignore -n option if provided)', required=False, default='/home/adrian/Desktop/03001627/image_embedding_03001627.caffemodel')
# '/home/adrian/Desktop/03001627/image_embedding_03001627.prototxt'
# '/media/adrian/Datasets/datasets/image_embedding/image_embedding_testing_03001627_rcnn/image_embedding_rcnn.prototxt'
parser.add_argument('--prototxt', '-p', help='Path to prototxt (if not at the default place)', required=False, default='/home/adrian/Desktop/03001627/image_embedding_03001627.prototxt')
parser.add_argument('--gpu_index', help='GPU index (default=0).', type=int, default=0)
parser.add_argument('--top_k', help='Retrieve top K shapes.', type=int, default=32)
args = parser.parse_args()

if args.caffemodel:
    image_embedding_caffemodel = args.caffemodel
if args.prototxt:
    image_embedding_prototxt = args.prototxt


part_id = 4
print 'My training'
# My training
# g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part' + str(part_id) + '.txt'  # Is correct
# image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn.prototxt'  # Is correct
# image_embedding_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold.prototxt'
# image_embedding_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_iter_40000.caffemodel'


# My Single Part Manifold (Part X)
g_shape_embedding_space_file_txt = '/media/adrian/Datasets/datasets/shape_embedding/part_shape_embedding_space_03001627_part' + str(part_id) + '.txt'  # Is correct
image_embedding_prototxt = os.path.join(g_part_image_semSeg_embedding_testing_folder, 'image_embedding_manifoldNet_part' + str(part_id) + '.prototxt')
image_embedding_caffemodel = os.path.join(g_part_image_semSeg_embedding_testing_folder, 'stacked_03001627_part' + str(part_id) + '_iter_400000.caffemodel')


print 'Computing image embedding for %s...' % args.image

image_embedding_array = extract_cnn_features(img_filelist=args.image,
                                             img_root='/',
                                             prototxt=image_embedding_prototxt,
                                             caffemodel=image_embedding_caffemodel,
                                             feat_name='image_embedding_part' + str(part_id),
                                             caffe_path=g_caffe_install_path,
                                             mean_file=g_mean_file)
image_embedding = image_embedding_array[0]


print 'Loading shape embedding space from %s...'%(g_shape_embedding_space_file_txt)
shape_embedding_space = [np.array([float(value) for value in line.strip().split(' ')]) for line in open(g_shape_embedding_space_file_txt, 'r')]
assert(image_embedding.size == shape_embedding_space[0].size)

print 'Computing distances and ranking...'
sorted_distances = sorted([(math.sqrt(sum((image_embedding-shape_embedding)**2)), idx) for idx, shape_embedding in enumerate(shape_embedding_space)])
print sorted_distances[0:args.top_k]

print 'Loading shape list from %s'%(g_shape_list_file)
shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
assert(len(shape_list) == len(shape_embedding_space))

visualization_filename = os.path.splitext(args.image)[0]+'_retrieval.html'
print 'Saving visualization to %s...'%(visualization_filename)
visualization_template = os.path.join(BASE_DIR, 'image_semSeg_based_part_shape_retrieval.html')
shutil.copy(visualization_template, visualization_filename)
for line in fileinput.input(visualization_filename, inplace=True):
    line = line.replace('QUERY_IMAGE_FILENAME', os.path.split(args.image)[-1])
    sys.stdout.write(line)

retrieval_list = ''
for i in range(args.top_k):
    shape_idx = sorted_distances[i][1]
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
 """%(synset, md5_id, synset, md5_id, synset, md5_id, md5_id)


for line in fileinput.input(visualization_filename, inplace=True):
    line = line.replace('RETRIEVAL_LIST', retrieval_list)
    sys.stdout.write(line)
