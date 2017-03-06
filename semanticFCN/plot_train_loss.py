import caffe
import surgery, score

import numpy as np
import os
import json


path = 'shapenet-fcn32s/'

loss_file = path + 'train_loss.txt'

# Open the trainning loss file
with open(loss_file, 'r') as f:
    train_loss = json.load(f)

