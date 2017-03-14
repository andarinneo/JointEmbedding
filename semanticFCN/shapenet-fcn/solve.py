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


# weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
# weights = '../voc-fcn32s/fcn32s-heavy-pascal.caffemodel'
weights = '../voc-fcn8s-atonce/fcn8s-atonce-pascal.caffemodel'

weights = '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/snapshot/train_iter_100000.caffemodel'

# Generate list of image/segmentation files to use

# run matlab script "create_seg_file_list.m", it will generate 2 files under "dataset/"
# "seg_file_list.mat" and "image_list.mat"

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# # scoring
# train_val_test = '../../datasets/image_embedding/train_val_test.txt'
# with open(train_val_test, 'r') as f:
#     img_idxs_train, label_idxs_train, img_idxs_test, label_idxs_test, img_idxs_val, label_idxs_val = json.load(f)
# val = img_idxs_val

# plt.ion()

loss_file = 'train_loss.txt'
train_loss = []
train_step = []
step_counter = 100000  # 0
# Run the training
debug = True
for _ in range(300000):
    step_counter += 1
    solver.step(50)  # (4000)
    # score.seg_tests(solver, False, val, layer='score')
    train_loss.append(round(solver.net.blobs['loss'].data.flatten()[0]))
    train_step.append(step_counter)

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

        plt.subplot(3, 8, 1)
        plt.imshow(train_img)
        plt.subplot(3, 8, 2)
        plt.imshow(train_label, vmin=0, vmax=5)
        for i in range(0, 21):
            heat_map = probability_maps[0, i, ...]
            plt.subplot(3, 8, (i+1)+2)  # the +2 is the offset for the image and the labels
            plt.imshow(heat_map, vmin=min_score, vmax=max_score)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # plt.show()
        plt.pause(1)

    # Save training loss
    with open(loss_file, 'w+') as f:
        json.dump([train_step, train_loss], f)

