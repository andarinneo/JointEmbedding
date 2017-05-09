import caffe
import numpy as np
import os
import json
import matplotlib.pyplot as plt


# Part manifold 1
base_prototxt = '/home/adrian/JointEmbedding/datasets/image_embedding/part_image_embedding_training_03001627_rcnn/batch_solvers/solver2.prototxt'
base_caffemodel = '/home/adrian/JointEmbedding/datasets/image_embedding/bvlc_reference_rcnn_ilsvrc13.caffemodel'


# init training
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(base_prototxt)
solver.net.copy_from(base_caffemodel)


# Run the training
for _ in range(100000):
    solver.step(1)  # (4000)

    # Test the distances
    # gt_coord1 = solver.net.blobs['image_embedding'].data[0]
    # est_coord1 = solver.net.blobs['shape_embedding'].data[0, :, 0, 0]

    gt_coord1 = solver.net.blobs['image_embedding_part1'].data[0]
    est_coord1 = solver.net.blobs['shape_embedding_part1'].data[0, :, 0, 0]

    dist = np.linalg.norm(gt_coord1 - est_coord1)

    aux1 = gt_coord1 - est_coord1
    aux2 = np.multiply(aux1, aux1)
    aux3 = np.sum(aux2)
    loss = aux3/(1*2)

    # caffe_loss = solver.net.blobs['embedding_loss'].data

    caffe_loss = solver.net.blobs['embedding_loss_part1'].data


    correct = caffe_loss - loss
    # END test the distances


