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

## load data
DATASET = 'IIAI-Body-Emotion'
ROOT_DIR = '/Users/cjx/PycharmProjects/bodyemotion'
path_pose_tr = op.join(ROOT_DIR, 'dataset', DATASET, "train.txt")
pose_tr = dataset.import_train(op.join(ROOT_DIR, 'dataset'), path_pose_tr)
pose_train = model.train(pose_tr)



