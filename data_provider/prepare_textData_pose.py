import os
import os.path as op
import glob


dataset ='IIAI-Body-Emotion'
root_data = op.abspath('..')
print(root_data)
WORD2CLASS={'NE': 0,
            'AN': 1,
            'FE': 2,
            'SA': 3,
            'HA': 4}

folders_seq = os.listdir( op.join(root_data, 'dataset', dataset) )
numb_seq = len(folders_seq)
with open( op.join( root_data, 'dataset', dataset, 'train.txt' ), 'w' ) as fout:
    for i in range(numb_seq):
        for joint_file in glob.glob(op.join(root_data, 'dataset', dataset, folders_seq[i], '*.npy')):
            joint_file = joint_file.split('/')
            word_emotion = joint_file[-1][3:5]
            label = WORD2CLASS[word_emotion]
            path = op.join( joint_file[-3],joint_file[-2],joint_file[-1] )
            fout.write( '{}\t{}\n'.format(path, label) )
fout.close()
