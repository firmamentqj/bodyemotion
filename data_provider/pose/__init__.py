import os
import os.path as op
import numpy as np

# This file builds a dataset iterator for training and test
# IIAI-Tony-Robbins-Body-Language Project
# Written by Jiaxin Chen

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
        pose = np.reshape( pose, [pose.shape[0]*pose.shape[1]] )
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





