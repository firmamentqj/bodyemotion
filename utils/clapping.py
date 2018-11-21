import numpy as np
import os
from configs import config as opts

import matplotlib.pyplot as plt

def plot2DCurve(x,y, frame_indx_status_on=None):
    plt.title( 'Pose-Emotion-Analysis' )

    plt.xlabel('Frame index')
    plt.ylabel('Distance between two hands')
    plt.plot(x,y,'bo-')
    for i in range(len(frame_indx_status_on)):
        plt.plot(frame_indx_status_on[i],y[frame_indx_status_on[i]],'go')
    plt.show()

def is_clapping(joints, opts, mode='static'):

    if mode=='static':
        numb_joints = joints.shape[0]
        dist = np.sqrt((joints[4][0:1]-joints[7][0:1])*(joints[4][0:1]-joints[7][0:1]))


        flag_clapping=dist<opts.threshold_clapping
        return dist, flag_clapping
###################
## main function ##
###################
## set parameters
#opts = []
#opts.threshold_clapping= 0.05

## load data
ROOT_DIR = '/home/jiaxinchen/Project/bodyemotion'
avg_joints = np.load( os.path.join(ROOT_DIR, 'clapping.npy') )
numb_frames = avg_joints.shape[0]
numb_joints = avg_joints.shape[1]

frame_indx = []
dist_hands = []
flag = []
frame_indx_clapping = []
for i in range(numb_frames):
    joint = avg_joints[i]
    temp_dist, temp_flag = is_clapping(joint, opts)
    frame_indx.append(i)
    dist_hands.append(temp_dist)
    if( temp_flag==True ):
        frame_indx_clapping.append(i)

plot2DCurve(frame_indx, dist_hands, frame_indx_clapping)
print(0)








