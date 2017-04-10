import caffe
import surgery, score
import numpy as np
import os
import json
import matplotlib.pyplot as plt


stacked_prototxt = 'train-manifold.prototxt'
stacked_caffemodel = 'snapshot/initial_stacked.caffemodel'

base_prototxt = '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/train-fcn8-atonce-5channels.prototxt'
base_caffemodel = '/home/adrian/JointEmbedding/semanticFCN/shapenet-fcn/snapshot/train_iter_100.caffemodel'


if not os.path.isfile(stacked_caffemodel):
    vgg_prototxt = '/home/adrian/JointEmbedding/semanticFCN/ilsvrc-nets/VGG_ILSVRC_16_layers.prototxt'
    vgg_caffemodel = '/home/adrian/JointEmbedding/semanticFCN/ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
    # Load weights into the complete architecture
    surgery.merge_caffe_models(base_prototxt = base_prototxt,
                               base_model = base_caffemodel,
                               top_prototxt = vgg_prototxt,
                               top_model = vgg_caffemodel,
                               stacked_prototxt = stacked_prototxt,
                               stacked_model = stacked_caffemodel)

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
for _ in range(1):
    step_counter += 1
    solver.step(1)  # (4000)
    # score.seg_tests(solver, False, val, layer='score')
    train_loss.append(round(solver.net.blobs['loss'].data.flatten()[0]))
    train_step.append(step_counter)

    # Save training loss
    with open(loss_file, 'w+') as f:
        json.dump([train_step, train_loss], f)

