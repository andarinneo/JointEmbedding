import caffe

import numpy as np
import random
from PIL import Image
import os.path
from sklearn.cross_validation import train_test_split
import json
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath('/home/adrian/JointEmbedding/src'))
from global_variables import *


class SHAPENETSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the rendered training images.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - shapenet_dir: path to image embedding `datasets/image_embedding` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset", mean=(104.00698793, 116.66876762, 122.67891434), split="valid")
        """

        # config
        params = eval(self.param_str)
        self.shapenet_dir = params['shapenet_dir']
        self.img_list = params['img_list']
        self.label_list = params['label_list']
        self.batch_size = params['batch_size']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.rescale_image = bool(params['rescale_image'])
        self.rescale_size = np.array(params['rescale_size'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)


        if 'train' in self.split:
            with open(g_syn_images_filelist_train, 'r') as f: self.img_indices = f.read().splitlines()
            with open(g_syn_labels_filelist_train, 'r') as f: self.label_indices = f.read().splitlines()
            self.idx = 0
        elif 'val' in self.split:
            with open(g_syn_images_filelist_val, 'r') as f: self.img_indices = f.read().splitlines()
            with open(g_syn_labels_filelist_val, 'r') as f: self.label_indices = f.read().splitlines()
            self.idx = 0
        elif 'test' in self.split:  # At this point (val=test), need to modify this ater debugging
            with open(g_syn_images_filelist_val, 'r') as f: self.img_indices = f.read().splitlines()
            with open(g_syn_labels_filelist_val, 'r') as f: self.label_indices = f.read().splitlines()
            self.idx = 0


        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.img_indices)-1)


    def reshape(self, bottom, top):
        if self.batch_size <= 1:
            # load image + label image pair
            self.data = self.load_image(self.idx)
            self.label = self.load_label(self.idx)
            # reshape tops to fit (leading 1 is for batch dimension)
            top[0].reshape(1, *self.data.shape)
            top[1].reshape(1, *self.label.shape)
        else:
            self.data = []
            self.label = []

            for i in range(0, self.batch_size, 1):
                self.idx = random.randint(0, len(self.img_indices) - 1)
                # load image + label image pair
                self.data.append(self.load_image(self.idx))
                self.label.append(self.load_label(self.idx))

            # reshape tops to fit (leading 1 is for batch dimension)
            top[0].reshape(self.batch_size, *self.data[0].shape)
            top[1].reshape(self.batch_size, *self.label[0].shape)


    def forward(self, bottom, top):
        if self.batch_size <= 1:
            # assign output
            top[0].data[...] = self.data
            top[1].data[...] = self.label
        else:
            for i in range(0, self.batch_size, 1):
                top[0].data[i, ...] = self.data[i]
                top[1].data[i, ...] = self.label[i]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.img_indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.img_indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.img_indices[idx])

        if self.rescale_image:
            im = im.resize(self.rescale_size, Image.BICUBIC)

        in_ = np.array(im, dtype=np.float32)

        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open(self.label_indices[idx])
        if self.rescale_image:
            im = im.resize(self.rescale_size, Image.NEAREST)

        width, height = im.size
        label = np.zeros([height, width], dtype=np.uint8)

        red, green, blue, alpha = im.split()
        red_ = np.array(red, dtype=np.float32)
        green_ = np.array(green, dtype=np.float32)
        blue_ = np.array(blue, dtype=np.float32)
        alpha_ = np.array(alpha, dtype=np.float32)

        # This only applies to chairs, (fix once chairs work)
        # Parts: (1, armrests, green), (2, back, red), (3, legs, blue), (4, seat, black)

        # Green
        aux = np.logical_and(np.logical_and(np.logical_and([red_ < 25], [green_ > 240]), [blue_ < 25]), [alpha_ > 240])
        mask1 = np.asarray(aux[0])
        label[mask1] = 1

        # Red
        aux = np.logical_and(np.logical_and(np.logical_and([red_ > 240], [green_ < 25]), [blue_ < 25]), [alpha_ > 240])
        mask2 = np.asarray(aux[0])
        label[mask2] = 2

        # Blue
        aux = np.logical_and(np.logical_and(np.logical_and([red_ < 25], [green_ < 25]), [blue_ > 240]), [alpha_ > 240])
        mask3 = np.asarray(aux[0])
        label[mask3] = 3

        # Black
        aux = np.logical_and(np.logical_and(np.logical_and([red_ < 25], [green_ < 25]), [blue_ < 25]), [alpha_ > 240])
        mask4 = np.asarray(aux[0])
        label[mask4] = 4

        # plt.imshow(mask4, aspect="auto")
        # plt.show()

        label = label[np.newaxis, ...]
        return label


class SHAPENETManifoldDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the rendered training images.

    Use this to feed data to a fully convolutional network + manifold output.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - shapenet_dir: path to image embedding `datasets/image_embedding` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset", mean=(104.00698793, 116.66876762, 122.67891434), split="valid")
        """

        # config
        params = eval(self.param_str)
        self.shapenet_dir = params['shapenet_dir']
        self.img_list = params['img_list']
        self.label_list = params['label_list']
        self.batch_size = params['batch_size']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.rescale_image = bool(params['rescale_image'])
        self.rescale_size = np.array(params['rescale_size'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)


        self.manifold = []
        for part_id in range(1, g_n_parts+1):
            manifold_name = g_part_shape_embedding_space_file_txt[:-4] + '_part' + str(part_id) + '.txt'
            self.manifold.append(np.loadtxt(manifold_name))


        if 'train' in self.split:
            with open(g_syn_images_filelist_train, 'r') as f: self.img_indices = f.read().splitlines()
            with open(g_syn_labels_filelist_train, 'r') as f: self.label_indices = f.read().splitlines()
            with open(g_syn_images_imageid2shapeid_train, 'r') as f: self.imageid2shapeid = f.read().splitlines()
            self.imageid2shapeid = map(int, self.imageid2shapeid)
            self.idx = 0
        elif 'val' in self.split:
            with open(g_syn_images_filelist_val, 'r') as f: self.img_indices = f.read().splitlines()
            with open(g_syn_labels_filelist_val, 'r') as f: self.label_indices = f.read().splitlines()
            with open(g_syn_images_imageid2shapeid_val, 'r') as f: self.imageid2shapeid = f.read().splitlines()
            self.imageid2shapeid = map(int, self.imageid2shapeid)
            self.idx = 0
        elif 'test' in self.split:  # At this point (val=test), need to modify this after debugging
            with open(g_syn_images_filelist_val, 'r') as f: self.img_indices = f.read().splitlines()
            with open(g_syn_labels_filelist_val, 'r') as f: self.label_indices = f.read().splitlines()
            with open(g_syn_images_imageid2shapeid_val, 'r') as f: self.imageid2shapeid = f.read().splitlines()
            self.imageid2shapeid = map(int, self.imageid2shapeid)
            self.idx = 0


        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.img_indices)-1)


    def reshape(self, bottom, top):
        if self.batch_size <= 1:
            # load image + label image pair
            self.data = self.load_image(self.idx)
            self.manifold_coord1 = self.load_manifold_coord(self.idx, 1)
            self.manifold_coord2 = self.load_manifold_coord(self.idx, 2)
            self.manifold_coord3 = self.load_manifold_coord(self.idx, 3)
            self.manifold_coord4 = self.load_manifold_coord(self.idx, 4)
            # reshape tops to fit (leading 1 is for batch dimension)
            top[0].reshape(1, *self.data.shape)
            top[1].reshape(1, *self.manifold_coord3.shape)

            # top[1].reshape(1, *self.manifold_coord1.shape)
            # top[2].reshape(1, *self.manifold_coord2.shape)
            # top[3].reshape(1, *self.manifold_coord3.shape)
            # top[4].reshape(1, *self.manifold_coord4.shape)
        else:
            self.data = []
            self.label = []
            self.manifold_coord1 = []
            self.manifold_coord2 = []
            self.manifold_coord3 = []
            self.manifold_coord4 = []

            for i in range(0, self.batch_size, 1):
                self.idx = random.randint(0, len(self.img_indices) - 1)
                # load image + label image pair
                self.data.append(self.load_image(self.idx))
                self.manifold_coord1.append(self.load_manifold_coord(self.idx, 1))
                self.manifold_coord2.append(self.load_manifold_coord(self.idx, 2))
                self.manifold_coord3.append(self.load_manifold_coord(self.idx, 3))
                self.manifold_coord4.append(self.load_manifold_coord(self.idx, 4))

            # reshape tops to fit (leading 1 is for batch dimension)
            top[0].reshape(self.batch_size, *self.data[0].shape)
            top[1].reshape(self.batch_size, *self.manifold_coord3[0].shape)

            # top[1].reshape(self.batch_size, *self.manifold_coord1[0].shape)
            # top[2].reshape(self.batch_size, *self.manifold_coord2[0].shape)
            # top[3].reshape(self.batch_size, *self.manifold_coord3[0].shape)
            # top[4].reshape(self.batch_size, *self.manifold_coord4[0].shape)


    def forward(self, bottom, top):
        if self.batch_size <= 1:
            # assign output
            top[0].data[...] = self.data
            top[1].data[...] = self.manifold_coord3

            # top[1].data[...] = self.manifold_coord1
            # top[2].data[...] = self.manifold_coord2
            # top[3].data[...] = self.manifold_coord3
            # top[4].data[...] = self.manifold_coord4
        else:
            for i in range(0, self.batch_size, 1):
                top[0].data[i, ...] = self.data[i]
                top[1].data[i, ...] = self.manifold_coord3[i]

                # top[1].data[i, ...] = self.manifold_coord1[i]
                # top[2].data[i, ...] = self.manifold_coord2[i]
                # top[3].data[i, ...] = self.manifold_coord3[i]
                # top[4].data[i, ...] = self.manifold_coord4[i]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.img_indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.img_indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.img_indices[idx])

        if self.rescale_image:
            im = im.resize(self.rescale_size, Image.BICUBIC)

        in_ = np.array(im, dtype=np.float32)

        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open(self.label_indices[idx])
        if self.rescale_image:
            im = im.resize(self.rescale_size, Image.NEAREST)

        width, height = im.size
        label = np.zeros([height, width], dtype=np.uint8)

        red, green, blue, alpha = im.split()
        red_ = np.array(red, dtype=np.float32)
        green_ = np.array(green, dtype=np.float32)
        blue_ = np.array(blue, dtype=np.float32)
        alpha_ = np.array(alpha, dtype=np.float32)

        # This only applies to chairs, (fix once chairs work)
        # Parts: (1, armrests, green), (2, back, red), (3, legs, blue), (4, seat, black)

        # Green
        aux = np.logical_and(np.logical_and(np.logical_and([red_ < 25], [green_ > 240]), [blue_ < 25]), [alpha_ > 240])
        mask1 = np.asarray(aux[0])
        label[mask1] = 1

        # Red
        aux = np.logical_and(np.logical_and(np.logical_and([red_ > 240], [green_ < 25]), [blue_ < 25]), [alpha_ > 240])
        mask2 = np.asarray(aux[0])
        label[mask2] = 2

        # Blue
        aux = np.logical_and(np.logical_and(np.logical_and([red_ < 25], [green_ < 25]), [blue_ > 240]), [alpha_ > 240])
        mask3 = np.asarray(aux[0])
        label[mask3] = 3

        # Black
        aux = np.logical_and(np.logical_and(np.logical_and([red_ < 25], [green_ < 25]), [blue_ < 25]), [alpha_ > 240])
        mask4 = np.asarray(aux[0])
        label[mask4] = 4

        # plt.imshow(mask4, aspect="auto")
        # plt.show()

        label = label[np.newaxis, ...]
        return label


    def load_manifold_coord(self, idx, part_id):
        """
        Load coordinates for each of the part 3D shape manifolds
        """
        md5_id = self.imageid2shapeid[idx]

        coordinates = []
        coordinates.append(self.manifold[part_id-1][md5_id-1])

        coordinates = np.array(self.manifold[part_id-1][md5_id-1], dtype=np.float32)

        return coordinates


