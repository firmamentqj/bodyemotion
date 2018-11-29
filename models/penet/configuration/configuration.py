import os
import os.path as op

WORD2CLASS={'NE': 0,
            'AN': 1,
            'FE': 2,
            'SA': 3,
            'HA': 4}
CLASS2WORD = {0: 'Neutral',
              1: 'Angry',
              2: 'Fearful',
              3: 'Sad',
              4: 'Happy'}

dataset = 'IIAI-Body-Emotion'
method = 'penet'
numb_class = len(WORD2CLASS)
## path
## root_dir = '/Users/cjx/PycharmProjects/bodyemotion'
root_dir = '/Users/nigel/PycharmProjects/tf-openpose'
exp_dir = op.join( root_dir, 'trainedModels', method )

## for training
network_name = 'PENET'
input_dim = 36
is_training = True
is_reuse = False
batch_size = 64
val_batch_size = 256
epochs = 10000
learning_rate = 1e-2
decay_step = 4000
decay_factor = 0.99
val_freq = 1000
save_checkpoint_freq = 1000
save_checkpoint_dir = exp_dir

## for testing
checkpoint_test_name = 'checkpoint-440000'