#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import lmdb
import shutil
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

shuffled_image_indexes = [int(line.strip()) for line in open(g_shuffled_image_indexes, 'r')]
train_val_split = [int(line.strip()) for line in open(g_syn_images_train_val_split, 'r')]

# It loads the pool5 feature descriptors for each train/val image and creates a DB using that
env = lmdb.open(g_pool5_lmdb, readonly=True)
if os.path.exists(g_pool5_lmdb_train):
    shutil.rmtree(g_pool5_lmdb_train)
    print 'purge the witch...'
env_train = lmdb.open(g_pool5_lmdb_train, map_size=int(1e12))
if os.path.exists(g_pool5_lmdb_val):
    shutil.rmtree(g_pool5_lmdb_val)
    print 'purge the witch...'
env_val = lmdb.open(g_pool5_lmdb_val, map_size=int(1e12))


cache_train = dict()
cache_val = dict()
txn_commit_count = 512

# shuffled_image_indexes = shuffled_image_indexes[0:20]  # BORRAR

report_step = 10000
with env.begin() as txn:
    for counter in range(len(shuffled_image_indexes)):
        idx = shuffled_image_indexes[counter]
        train_val = train_val_split[counter]

        key = '{:0>10d}'.format(idx)
        value = bytes(txn.get(key))

        if train_val == 1:
            cache_train[key] = value
        elif train_val == 0:
            cache_val[key] = value
            
        if len(cache_train) == txn_commit_count or counter == len(shuffled_image_indexes)-1:
            with env_train.begin(write=True) as txn_train:
                for k, v in sorted(cache_train.iteritems()):
                    txn_train.put(k, v)
            cache_train.clear()
        if len(cache_val) == txn_commit_count or counter == len(shuffled_image_indexes)-1:
            with env_val.begin(write=True) as txn_val:
                for k, v in sorted(cache_val.iteritems()):
                    txn_val.put(k, v)
            cache_val.clear()
                
        if (counter % report_step) == 0:
            print datetime.datetime.now().time(), '-', counter, 'of', len(shuffled_image_indexes), 'processed!'
        # counter += 1  # this was left behind, i commented it out, looks very wrong... (?)

env.close()
env_train.close()
env_val.close()
