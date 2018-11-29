import tensorflow as tf
import os
import os.path as op
import numpy as np
import time
from signal import SIGINT, SIGTERM
import sys
from utils import lbtoolbox as lb
from datetime import timedelta
import math

from models.poseEmoNet import PEMLPNET as model
from utils.input_data import buildTrainData
import data_provider.pose as dataset
import models.penet as model
from models.penet.configuration import configuration as config

## load data
numb_splits = 10
DATASET = config.dataset
ROOT_DIR = config.root_dir

acc_avg_validation = 0
for i in range( numb_splits ):
    print('\n==============================RUNNING SPLIT %02d=============================='%(i+1))
    flag_reuse = i!=0
    path_pose_tr = op.join(ROOT_DIR, 'data', DATASET, "train_split"+'%02d'%(i)+'.txt')
    path_pose_val = op.join(ROOT_DIR, 'data', DATASET, "validation_split"+'%02d'%(i)+'.txt')
    pose_tr = dataset.import_train(op.join(ROOT_DIR, 'dataset'), path_pose_tr)
    pose_val = dataset.import_validation(op.join(ROOT_DIR, 'dataset'), path_pose_val)

    acc_validation = model.train_val(pose_tr, pose_val, is_reuse=flag_reuse)
    acc_avg_validation += acc_validation

acc_avg_validation = acc_avg_validation / numb_splits
print( 'Mean Cross-validation Classification Accuracy for Pose-Emotion Estimation: %.02f'%(100*acc_avg_validation) )




