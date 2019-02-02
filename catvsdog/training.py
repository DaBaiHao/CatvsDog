#%%
import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%
N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 15000 # suggest >10k
learing_rate = 0.0001

#%%
def run_training():
    train_dir = 'train'
    logs_train_dir = 'logs/train'
    
    train, train_label = input_data.gey_files(train_dir)


