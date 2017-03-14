import caffe
import surgery, score

import numpy as np
import os
import json
import matplotlib.pyplot as plt


path = 'shapenet-fcn/'

loss_file = path + 'train_loss.txt'

# Open the trainning loss file
with open(loss_file, 'r') as f:
    [train_step, train_loss] = json.load(f)

plt.plot(train_step, train_loss)
plt.show()




