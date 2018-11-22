import os
import os.path as op
import numpy as np

# This file builds a dataset iterator for training and test
# Inception-Body-Language Project
# Write by Jiaxin Chen

class Dataset():
    def __init__(self, data_root, path, is_train=True):
        self.lines = open(path, 'r').readlines()
        self.data_root = data_root
        self.n_samples = len(self.lines)
        self.is_train = is_train
        self.img = [0]*self.n_samples
        self.pose = [0]*self.n_samples
        self.label = [0]*self.n_samples
    def get_pose(self,i):
        path = op.join(self.data_root, self.line[i].strip().split()[]





