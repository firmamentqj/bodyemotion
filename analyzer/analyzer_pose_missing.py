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
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from models.poseEmoNet import PEMLPNET as model
from utils.input_data import buildTrainData
import data_provider.pose as dataset
import models.penet as model
from models.penet.configuration import configuration as config
from models.penet.input import Dataset

'''
def computeMissingFreq(pose_datasdt, batch_size):
    numb_batches = int( pose_tr.n_samples / batch_size )
    counter = 0
    for i in range( numb_batches ):
        poses, labels = pose_tr.next_batch( config.batch_size )
        poses = np.reshape( poses, [poses.shape[0], 18,2] )
        for jj in range(poses.shape[0]):
            counter+=1
            pose = np.squeeze(poses[jj])
            pose = np.sum(pose*pose,1)
            indicator_pose_missing = np.array(pose<1e-10, dtype=float)

            if i == 0 and jj == 0:
                stats_missing = indicator_pose_missing
            else:
                stats_missing += indicator_pose_missing
    print(counter)
    print(pose_tr.n_samples)

    return (stats_missing / counter)*100
## load data
numb_splits = 10
DATASET = config.dataset
ROOT_DIR = config.root_dir

acc_avg_validation = 0
for i in range( 1 ):
    print('\n==============================RUNNING SPLIT %02d=============================='%(i+1))
    flag_reuse = i!=0
    path_pose_tr = op.join(ROOT_DIR, 'data', DATASET, "train_split"+'%02d'%(i)+'.txt')
    path_pose_val = op.join(ROOT_DIR, 'data', DATASET, "validation_split"+'%02d'%(i)+'.txt')
    pose_tr = dataset.import_train(op.join(ROOT_DIR, 'dataset'), path_pose_tr)
    pose_val = dataset.import_validation(op.join(ROOT_DIR, 'dataset'), path_pose_val)
    pose_tr = Dataset(pose_tr)
    pose_val = Dataset(pose_val)

    print('For training data')
    stats_missing_tr = computeMissingFreq(pose_tr, config.batch_size)
    plt.bar( np.arange(stats_missing_tr.shape[0]), stats_missing_tr )
    str_xticks_tr = map(lambda i: '%d'%(i), np.arange(stats_missing_tr.shape[0]))
    plt.xticks( np.arange(stats_missing_tr.shape[0]), str_xticks_tr )
    plt.xlabel('Number of Joint')
    plt.ylabel('Percentage of Value Missing (%)')
    plt.grid()
    plt.show()
'''
def calAngle( origin, start_point, end_point ):
    if( (origin[0]<1e-14 and origin[1]<1e-14) or (start_point[0]<1e-14 and start_point[1]<1e-14)  or (end_point[0]<1e-14 and end_point[1]<1e-14) ):
        return -1
    vecA = start_point-origin
    vecB = end_point-origin
    lenA = np.sqrt(np.sum(vecA ** 2))
    lenB = np.sqrt(np.sum(vecB ** 2))
    det = vecA[1]*vecB[0]-vecA[0]*vecB[1]

    if(lenA==0) or (lenB==0) :
        return 0
    elif( det<0 ):
        cos = np.dot(vecA, vecB) / ((lenA * lenB))
        return 2*math.pi - math.acos( cos )
    else:
        cos = np.dot( vecA,vecB ) / ((lenA*lenB))
        return math.acos(cos)


origin = np.array([0.1,0.1])
start_point = np.array([-0.1,-0.1])+origin
end_point = np.array([0,-1])+origin
plt.plot( np.array([origin[0],start_point[0]]), np.array([origin[1],start_point[1]]), 'b>-' )
plt.plot( np.array([origin[0],end_point[0]]), np.array([origin[1],end_point[1]]), 'g>-' )
angle = calAngle( origin, start_point, end_point )
plt.xlabel('%s degrees'%(str(int(math.degrees(angle)))))
print( '{}'.format(angle) )
plt.show()
