import caffe
import surgery, score
import numpy as np
import os
import json
import matplotlib.pyplot as plt

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass


base_prototxt = 'train-fcn8-atonce.prototxt'
# base_model = '../voc-fcn8s-atonce/fcn8s-atonce-pascal.caffemodel'
base_model = '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/snapshot/train_iter_160000.caffemodel'

adapted_prototxt = 'train-fcn8-atonce-5channels.prototxt'
# adapted_model = 'snapshot/base_5channels.caffemodel'
adapted_model = 'snapshot/train_iter_40000.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

# reduce number of channels of original FCN
if not os.path.isfile(adapted_model):
    surgery.reduce_fcn_dimension(base_prototxt,
                                 base_model,
                                 adapted_prototxt,
                                 adapted_model)

# load network
solver = caffe.SGDSolver('solver-5channels.prototxt')
solver.net.copy_from(adapted_model)

# # surgeries
# interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
# surgery.interp(solver.net, interp_layers)

# # scoring
# train_val_test = '../../datasets/image_embedding/train_val_test.txt'
# with open(train_val_test, 'r') as f:
#     img_idxs_train, label_idxs_train, img_idxs_test, label_idxs_test, img_idxs_val, label_idxs_val = json.load(f)
# val = img_idxs_val

plt.ion()

loss_file = 'train_loss.txt'
train_loss = []
train_step = []
step_counter = 0  # 0
# Run the training
debug = True
for _ in range(300000):
    step_counter += 1
    solver.step(50)  # (4000)
    # score.seg_tests(solver, False, val, layer='score')
    train_loss.append(round(solver.net.blobs['loss'].data.flatten()[0]))

    # DEBUG SCORES AS HEAT MAPS
    if debug:
        # in_ = in_[:, :, ::-1]
        # in_ -= self.mean
        # in_ = in_.transpose((2, 0, 1))

        # FIX THE RECONVERSION TO NORMAL RGB VALUES!!!!
        train_img = solver.net.blobs['data'].data[0, ...].transpose((1, 2, 0)) + (104.00699, 116.66877, 122.67892)

        train_label = solver.net.blobs['label'].data[0, 0, ...]

        probability_maps = solver.net.blobs['score'].data
        max_score = probability_maps.max()
        min_score = probability_maps.min()

        plt.subplot(2, 4, 1)
        plt.imshow(train_img)
        plt.subplot(2, 4, 2)
        plt.imshow(train_label, vmin=0, vmax=5)
        for i in range(0, 5):  # original maps are from 0:21
            heat_map = probability_maps[0, i, ...]
            plt.subplot(2, 4, (i+1)+2)  # the +2 is the offset for the image and the labels
            plt.imshow(heat_map, vmin=min_score, vmax=max_score)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # plt.show()
        plt.pause(1)

    # Save training loss
    with open(loss_file, 'w+') as f:
        json.dump([train_step, train_loss], f)

