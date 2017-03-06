import caffe
import surgery, score

import numpy as np
import os
import json

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass


# weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
# weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
weights = '../voc-fcn32s/fcn32s-heavy-pascal.caffemodel'

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

# scoring
train_val_test = '../../datasets/image_embedding/train_val_test.txt'
with open(train_val_test, 'r') as f:
    img_idxs_train, label_idxs_train, img_idxs_test, label_idxs_test, img_idxs_val, label_idxs_val = json.load(f)
val = img_idxs_val

loss_file = 'train_loss.txt'
train_loss = []
# Run the training
for _ in range(100000):
    solver.step(5)  # (4000)
    # score.seg_tests(solver, False, val, layer='score')
    train_loss.append(solver.net.blobs['loss'].data.flatten()[0])

    with open(loss_file, 'w+') as f:
        json.dump(['train_loss'], f)

