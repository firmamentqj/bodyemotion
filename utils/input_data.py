import numpy as np
import os
import os.path as op

WORD2CLASS={'NE': 0,
            'AN': 1,
            'FE': 2,
            'SA': 3,
            'HA': 4}

def extractPoseRep( pose ):
    # implementation need to be updated
    return 0


## define read data
def read_pose_data( pose_path, numb_classes ):
    poses = list([])
    labels = list([])

    files_pose = os.listdir( pose_path )
    for filename in files_pose:
        #print('Process {}'.format(filename))
        pose = np.array(np.load(op.join(op.join(pose_path, filename))))
        pose = pose[:, 0:2]
        pose = pose.flatten()
        category = filename.split('.')
        category = category[0][3:5]
        label = WORD2CLASS[category]
        label_vec = np.zeros([numb_classes], dtype='int32')
        label_vec[label]=1

        if (np.sum(pose)>0):
            poses.append(pose)
            labels.append(list(label_vec))

    poses = np.array(poses)
    labels = np.array(labels)

    return poses, labels

class buildTrainData:
    def __init__(self, pose_path_train, numb_classes, batch_size ):
        poses, labels = read_pose_data(pose_path_train, numb_classes)
        self.datapath = pose_path_train
        self.numb_classes = numb_classes
        self.poses = np.array(poses)
        self.labels = np.array(labels)
        self.batch_size = batch_size

    def batches(self):
        randindx = np.arange(self.poses.shape[0])
        np.random.shuffle(randindx)
        poses = self.poses[randindx, ...]
        labels = self.labels[randindx, ...]
        numb_samples = self.poses.shape[0]
        b_poses = poses[0:self.batch_size, ...]
        b_labels = labels[0:self.batch_size, ...]
        #yield b_poses, b_labels

        for i in range(0, numb_samples, self.batch_size):
             b_poses = poses[i*self.batch_size:(i+1)*self.batch_size,...]
             b_labels = labels[i*self.batch_size:(i+1)*self.batch_size,...]
             yield b_poses, b_labels

class Dataset(object):
    def __init__(self, dataset, output_dim, code_dim):
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        self._codes = np.zeros((self.n_samples, code_dim), dtype=np.float32)
        self._triplets = np.array([])
        self._trip_index_in_epoch = 0
        self._index_in_epoch = 0
        self._epochs_complete = 0
        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        return

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            if self._train:
                self._epochs_complete += 1
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        data, label = self._dataset.data(self._perm[start:end])
        return data, label
