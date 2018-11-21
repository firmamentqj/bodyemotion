import tensorflow as tf
import os
import os.path as op
import numpy as np
import time
from signal import SIGINT, SIGTERM
import sys
from utils import lbtoolbox as lb
from datetime import timedelta

from models.poseEmoNet import PEMLPNET as model
from utils.input_data import buildTrainData

## Define global variables
ROOT_DIR = '.'
DATASET = 'BEAST'
NUMB_CLASS = 5

# train PENET
BATCH_SIZE_TRAIN = 64
DECAY_STAT_ITERATION = 10000
MAX_NUM_ITER = 100000
CHECKPOINT_FREQUENCY = 1000
LEARNING_RATE = 1e-3
MODELNAME = 'PEMLPNET'
RESUME = False
ROOT_EXP = op.join(ROOT_DIR, 'trainedModel', MODELNAME)

## build model
b_poses = tf.placeholder( 'float32', shape=(None, 18*2) )
b_labels = tf.placeholder( 'int32', shape=(None, 5) )

db_train = buildTrainData( op.join( ROOT_DIR, 'data', DATASET, 'poses' ), NUMB_CLASS, BATCH_SIZE_TRAIN )

if(MODELNAME=='PEMLPNET'):
    net = model.PEMLPNET(b_poses, NUMB_CLASS, is_training=True, is_reuse=False, name='PENET')
else:
    raise ValueError('Currently only support PEMLPNET.')
val_loss = model.loss_entropy( net['logits'], b_labels, NUMB_CLASS )

global_step = tf.Variable(0, name='global_step', trainable=False)
if 0 <= DECAY_STAT_ITERATION < MAX_NUM_ITER:
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE,
        tf.maximum(0, global_step - DECAY_STAT_ITERATION),
        MAX_NUM_ITER - DECAY_STAT_ITERATION, 0.001)
else:
        learning_rate = LEARNING_RATE
optimizer = tf.train.AdamOptimizer(learning_rate)
train_var = tf.trainable_variables()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = optimizer.minimize(val_loss, var_list=train_var, global_step=global_step)

init_op = tf.global_variables_initializer()
checkpoint_saver = tf.train.Saver(max_to_keep=0)

if not op.isdir(ROOT_EXP):
    try:
        os.makedirs(ROOT_EXP)
    except:
        pass
log_file = open(op.join(ROOT_EXP, "train.txt"), 'w')
## train model
with tf.Session() as sess_train:
    if RESUME:
        last_checkpoint = tf.train.latest_checkpoint(ROOT_EXP)
        log_file.write('Restoring parameters from {}.\n'.format(last_checkpoint))
        print('Restoring parameters from {}.\n'.format(last_checkpoint))
        checkpoint_saver.restore(sess_train, last_checkpoint)
    else:
        sess_train.run(init_op)
        checkpoint_saver.save(sess_train, ROOT_EXP, global_step=0)

    start_step = sess_train.run(global_step)
    log_file.write('Starting training from iteration {}.\n'.format(start_step))
    print('Starting training from iteration {}.\n'.format(start_step))
# ********************************
# *  run the training operation
# ********************************
    start_step = sess_train.run(global_step)
    print('Starting training from iteration {}.\n'.format(start_step))
    log_file.write('Starting training from iteration {}.\n'.format(start_step))
    with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
        step = 0
        flag_stop = False
        while (not flag_stop):
            for b_poses_, b_labels_ in db_train.batches():
                time_iter_start = time.time()
                feed_dict = {b_poses: b_poses_,
                             b_labels: b_labels_}
                _, b_loss, b_prediction_, b_labels_, step = sess_train.run([train_op, val_loss, net['prediction'], b_labels, global_step], feed_dict=feed_dict)
                time_elapsed = time.time() - time_iter_start

                acc=0
                b_correct_prediction = np.equal(np.argmax(b_prediction_, 1), np.argmax(b_labels_, 1))
                b_correct_prediction.astype("float32")
                b_correct_prediction = np.sum(b_correct_prediction)
                acc = b_correct_prediction / BATCH_SIZE_TRAIN

                seconds_todo = (MAX_NUM_ITER - step) * time_elapsed
                print('iter:{:6d}, loss={:.4f}  acc={:.2f}%  ETA: {} ({:.2f}s/it)'.format(
                    step,
                    b_loss,
                    acc*100,
                    timedelta(seconds=int(seconds_todo)),
                    time_elapsed))
                if (step % 200 == 0):
                    log_file.write('iter:{:6d}, loss={:.4f}  acc={:.4f}  ETA: {} ({:.2f}s/it)\n'.format(
                        step,
                        b_loss,
                        acc,
                        timedelta(seconds=int(seconds_todo)),
                        time_elapsed))
                if (CHECKPOINT_FREQUENCY > 0 and
                        step % CHECKPOINT_FREQUENCY == 0):
                    checkpoint_saver.save(sess_train, os.path.join(ROOT_EXP, 'checkpoint'), global_step=step)
                if u.interrupted:
                    log_file.write("Interrupted on request!\n")
                    log_file.close()
                    break
                if (step >= MAX_NUM_ITER):
                    flag_stop = True
                    break

## test model

## save data
