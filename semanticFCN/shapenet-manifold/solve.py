import caffe
import surgery, score
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



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


with open(loss_file, 'r') as f:
    [train_step, train_loss] = json.load(f)

# cutoff = 1500
# fs = 50000
# train_loss_smooth = butter_lowpass_filtfilt(train_loss, cutoff, fs)


plt.plot(train_step, train_loss)
plt.show()



# Run the training
debug = True
for _ in range(100000):
    step_counter += 1
    solver.step(50)  # (4000)
    # score.seg_tests(solver, False, val, layer='score')
    train_loss.append(round(solver.net.blobs['embedding_loss_part3'].data.flatten()[0]))
    train_step.append(step_counter)

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
    # with open(loss_file, 'w+') as f:
    #     json.dump([train_step, train_loss], f)

