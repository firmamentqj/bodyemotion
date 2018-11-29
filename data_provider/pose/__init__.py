import os
import os.path as op
import numpy as np
import math

# This file builds a dataset iterator for training and test
# IIAI-Tony-Robbins-Body-Language Project
# Written by Jiaxin Chen
def calAsymPairiedJointsDistance( matA, matB ):
    ## return a (12*11/2=66)-d vector containing distance information of an input pose
    dist = list([])
    for i in range(matA.shape[0]):
        for j in range(matB.shape[0]):
            if( i<j ):
                dist.append( np.sqrt(np.sum((matA[i]- matB[j])**2)) )

    return np.squeeze( np.array( dist ) )

def calAngle( origin, start_point, end_point ):
    ## return a real number in the range [0,2*pi]
    if( (origin[0]<1e-14 and origin[1]<1e-14) or (start_point[0]<1e-14 and start_point[1]<1e-14)  or (end_point[0]<1e-14 and end_point[1]<1e-14) ):
        return -1
    vecA = start_point-origin
    vecB = end_point-origin
    vecA.astype('float32')
    vecB.astype('float32')
    lenA = float(np.sqrt(np.sum(vecA ** 2)))
    lenB = float(np.sqrt(np.sum(vecB ** 2)))
    det = vecA[1]*vecB[0]-vecA[0]*vecB[1]

    if(lenA==0) or (lenB==0) :
        return 0
    elif( det<0 ):
        if not (lenA==0):
            vecA = vecA / (lenA+1e-8)
        if not (lenB==0):
            vecB = vecB / (lenB+1e-8)
        cos = np.dot(vecA, vecB)
        # print("Here 1")
        # print(cos)
        # print( "vecA=[{},{}],vecB=[{},{}]".format( vecA[0], vecA[1], vecB[0], vecB[1] ) )
        return 2*math.pi - math.acos( cos )
    else:
        if not (lenA==0):
            vecA = vecA / (lenA+1e-8)
        if not (lenB==0):
            vecB = vecB / (lenB+1e-8)
        cos = np.dot(vecA,vecB)
        # print("Here 2")
        # print(cos)
        return math.acos(cos)

def calJointsAngles( Joints ):
    ## return a 6-d vector containing angle information of an input pose
    angles = list([])
    angles.append(calAngle(Joints[1], Joints[0], Joints[5]))
    angles.append(calAngle(Joints[1], Joints[2], Joints[0]))
    angles.append(calAngle(Joints[2], Joints[1], Joints[3]))
    angles.append(calAngle(Joints[3], Joints[2], Joints[4]))
    angles.append(calAngle(Joints[5], Joints[6], Joints[1]))
    angles.append(calAngle(Joints[6], Joints[7], Joints[5]))

    return np.squeeze(np.array(angles))


class Dataset(object):
    def __init__(self, modal, data_root, path, train=True):
        self.lines = open(path, 'r').readlines()
        self.data_root = data_root
        self.n_samples = len(self.lines)
        self.train = train
        assert modal == 'pose'
        self.modal = 'pose'
        self._pose = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.pose_data
        self.all_data = self.pose_all_data

    def get_pose(self, i):
        path = op.join(self.data_root, self.lines[i].strip().split()[0])
        pose = np.load( path )
        pose = pose[...,0:2]
        ## extract partial poses from upper body
        pose = pose[[0,1,2,3,4,5,6,7,14,15,16,17]]
        ## additional information
        dist = calAsymPairiedJointsDistance( pose, pose )
        angles = calJointsAngles( pose )
        ## add additional information
        pose = np.squeeze(np.reshape(pose, [pose.shape[0] * pose.shape[1]]))
        #pose = np.concatenate((pose, dist, angles), axis=0)

        return pose

    def get_label(self, i):
        return int( self.lines[i].strip().split()[1] )

    def pose_data(self, indexes):
        if self._status:
            return (self._pose[indexes, :], self._label[indexes, :])
        else:
            ret_pose = []
            ret_label = []
            for i in indexes:
                try:
                    if self.train:
                        if not self._load[i]:
                            self._pose[i] = self.get_pose(i)
                            self._label[i] = self.get_label(i)
                            self._load[i] = 1
                            self._load_num += 1
                        ret_pose.append(self._pose[i])
                        ret_label.append(self._label[i])
                    else:
                        self._label[i] = self.get_label(i)
                        ret_pose.append(self.get_pose(i))
                        ret_label.append(self._label[i])
                except Exception as e:
                    print('cannot open {}, exception: {}'.format(self.lines[i].strip(), e))

            if self._load_num == self.n_samples:
                self._status = 1
                self._pose = np.asarray(self._pose)
                self._label = np.asarray(self._label)
            return np.asarray(ret_pose), np.asarray(ret_label)

    def pose_all_data(self):
        if self._status:
            return (self._pose, self._label)

    def get_labels(self):
        for i in range(self.n_samples):
            if self._label[i] is not list:
                self._label[i] = int( self.lines[i].strip().split()[1:] )
        return np.asarray(self._label)


def import_train(data_root, pose_tr):

    return (Dataset('pose', data_root, pose_tr, train=True))


def import_validation(data_root, pose_val ):

    return Dataset('pose', data_root, pose_val, train=False)





