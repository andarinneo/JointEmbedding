import os
import sys
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
# input: g_syn_images_bkg_overlaid_folder
#
# output: g_syn_images_filelist
#       Contains a list of all RGB train/val/test images
# output: g_syn_images_imageid2shapeid
#       Contains a list of all segmentation train/val/test label images
# output: g_syn_images_imageid2manifold
#       Contains a list of the corresponding manifold coordinates for all train/val/test images
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---

shape_md5_list = [line.strip().split(' ')[1] for line in open(g_shape_list_file, 'r')]
print len(shape_md5_list), 'shapes!'

# get all img and label files
img_shape_id_pairs = []  # full path
label_shape_id_pairs = []  # full path
for iterator_dir in g_shapenet_synset_set:
    # RGB images folder
    synset_dir = os.path.join(g_syn_images_bkg_overlaid_folder, iterator_dir)
    # Segmentation label
    segset_dir = os.path.join(g_syn_images_cropped_folder, iterator_dir)
    for shape_id, shape_md5 in enumerate(shape_md5_list):
        # RGB images folder
        shape_dir = os.path.join(synset_dir, shape_md5)
        # Segmentation label
        seg_dir = os.path.join(segset_dir, shape_md5)

        imgs = os.listdir(shape_dir)
        for img in imgs:
            label = img[:-4] + '_seg.png'
            img_shape_id_pairs.append((os.path.join(shape_dir, img), shape_id, os.path.join(seg_dir, label)))
print len(img_shape_id_pairs), ' syn images!'

# shuffle
random.seed(9527)  # seed random with a fixed number
img_shape_id_pairs = random.sample(img_shape_id_pairs, len(img_shape_id_pairs))

fout_filelist = open(g_syn_images_filelist, 'w')
fout_idmap = open(g_syn_images_imageid2shapeid, 'w')
fout_seglist = open(g_syn_labels_filelist, 'w')
for img_shape_id_pair in img_shape_id_pairs:
    fout_filelist.write('%s\n' % (img_shape_id_pair[0]))
    fout_idmap.write('%d\n' % (img_shape_id_pair[1]))
    fout_seglist.write('%s\n' % (img_shape_id_pair[2]))
fout_filelist.close()
fout_idmap.close()
fout_seglist.close()
