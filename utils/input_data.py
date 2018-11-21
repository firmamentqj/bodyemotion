import numpy as np
import os
import os.path as op

WORD2CLASS={'NE': 0,
            'AN': 1,
            'FE': 2,
            'SA': 3,
            'HA': 4}

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
        yield b_poses, b_labels

        # for i in range(0, numb_samples, self.batch_size):
        #     b_poses = poses[i*self.batch_size:(i+1)*self.batch_size,...]
        #     b_labels = labels[i*self.batch_size:(i+1)*self.batch_size,...]
        #     yield b_poses, b_labels