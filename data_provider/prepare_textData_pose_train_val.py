import os
import os.path as op
import glob
from models.penet.configuration import configuration as config
import math
import numpy as np

## load params
dataset = config.dataset
root_data = config.root_dir
WORD2CLASS=config.WORD2CLASS

train_val_ratio = 0.5
numb_splits = 10
## load data
folders_seq = os.listdir( op.join(root_data, 'dataset', dataset) )
numb_seq = len(folders_seq)

labels_full = list([])
paths_full = list([])
for i in range(numb_seq):
    for joint_file in glob.glob(op.join(root_data, 'dataset', dataset, folders_seq[i], '*.npy')):
        joint_file = joint_file.split('/')
        word_emotion = joint_file[-1][3:5]
        label = WORD2CLASS[word_emotion]
        path = op.join(joint_file[-3], joint_file[-2], joint_file[-1])
        labels_full.append(label)
        paths_full.append(path)

## split the full dataset into train dataset and validation data
numb_samples_full = len(labels_full)
for i in range( numb_splits ):
    print( 'Processing {}/{}-th split...'.format(i, numb_splits) )
    with open(op.join(root_data, 'data', dataset, 'train_split'+'%02d'%(i)+'.txt'), 'w') as fout_train, open(op.join(root_data, 'data', dataset, 'validation_split' + '%02d' % (i) + '.txt'), 'w') as fout_validation:
        unique_class = np.unique( np.array(labels_full) )
        for jj in range( len(unique_class) ):
            index_full = np.squeeze(np.array( np.where( labels_full==unique_class[jj] ) ))
            numb_full = len( index_full )
            numb_train = int( math.ceil( numb_full*train_val_ratio ) )
            numb_validation = numb_full-numb_train
            np.random.shuffle( index_full )
            index_train = index_full[0:numb_train]
            index_validation = index_full[numb_train:]
            for kkk in range( len( index_train ) ):
                fout_train.write('{}\t{}\n'.format(paths_full[index_train[kkk]], labels_full[index_train[kkk]]))
            for kkk in range( len( index_validation ) ):
                fout_validation.write('{}\t{}\n'.format(paths_full[index_validation[kkk]], labels_full[index_validation[kkk]]))
    fout_train.close()
    fout_validation.close()
