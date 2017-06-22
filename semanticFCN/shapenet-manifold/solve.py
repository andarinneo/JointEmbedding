import caffe
import surgery, score
import numpy as np
import os
import json
import matplotlib.pyplot as plt



stacked_prototxt = 'train-manifold-alexnet.prototxt'
# stacked_caffemodel = 'snapshot/initial_stacked.caffemodel'
stacked_caffemodel = 'snapshot/train-manifold_iter_30000.caffemodel'

sem_seg_prototxt = '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/deploy-fcn8-atonce-5channels.prototxt'
sem_seg_caffemodel = '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/snapshot/train_iter_60000.caffemodel'

manifold_prototxt = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/image_embedding_rcnn_single_manifold_part3.prototxt'
manifold_caffemodel = '/media/adrian/Datasets/datasets/image_embedding/part_image_embedding_testing_03001627_rcnn/snapshots_03001627_part1_iter_100000.caffemodel'


if not os.path.isfile(stacked_caffemodel):
    # Load weights into the complete architecture
    surgery.merge_FCN_AlexNet_models(base_prototxt=sem_seg_prototxt,
                                     base_model=sem_seg_caffemodel,
                                     top_prototxt=manifold_prototxt,
                                     top_model=manifold_caffemodel,
                                     stacked_prototxt=stacked_prototxt,
                                     stacked_model=stacked_caffemodel,
                                     layer_prefix='manifold')

# init training
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(stacked_caffemodel)


loss_file = 'train_loss.txt'
train_loss = []
train_step = []
step_counter = 0  # 0
# Run the training
debug = True
for _ in range(100000):
    step_counter += 1
    solver.step(1)  # (4000)
    # score.seg_tests(solver, False, val, layer='score')
    train_loss.append(round(solver.net.blobs['embedding_loss_part3'].data.flatten()[0]))
    train_step.append(step_counter)

    # DEBUG SCORES AS HEAT MAPS
    if debug:
        # in_ = in_[:, :, ::-1]
        # in_ -= self.mean
        # in_ = in_.transpose((2, 0, 1))

        # FIX THE RECONVERSION TO NORMAL RGB VALUES!!!!
        train_img = solver.net.blobs['data'].data[0, ...].transpose((1, 2, 0)) + (104.00699, 116.66877, 122.67892)

        probability_maps = solver.net.blobs['score'].data
        max_score = probability_maps.max()
        min_score = probability_maps.min()

        plt.subplot(2, 4, 1)
        plt.imshow(train_img)
        for i in range(0, 5):  # original maps are from 0:21
            heat_map = probability_maps[0, i, ...]
            plt.subplot(2, 4, (i+1)+1)  # the +2 is the offset for the image and the labels
            plt.imshow(heat_map, vmin=min_score, vmax=max_score)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # plt.show()
        plt.pause(1)


    # Test the distances
    gt_coord1 = solver.net.blobs['manifold_coord3'].data[0]
    # est_coord1 = solver.net.blobs['feat_fc8_part1'].data[0]
    est_coord1 = solver.net.blobs['image_embedding_part3'].data
    dist = np.linalg.norm(gt_coord1 - est_coord1)

    aux1 = gt_coord1 - est_coord1
    aux2 = np.multiply(aux1, aux1)
    aux3 = np.sum(aux2)
    loss = aux3/2

    caffe_loss = solver.net.blobs['embedding_loss_part3'].data

    correct = caffe_loss - loss
    # END test the distances

    # Save training loss
    with open(loss_file, 'w+') as f:
        json.dump([train_step, train_loss], f)

