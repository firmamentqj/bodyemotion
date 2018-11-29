import numpy as np
import tensorflow as tf
import os
import os.path as op
import random
import shutil
import time
from datetime import datetime
slim = tf.contrib.slim

'''
def network_architecture(input, numb_class, is_training, is_reuse, name=None):
    if not name == None:
        with tf.variable_scope(name):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net['input'] = input
            net['mlp1'] = slim.fully_connected(
                net['input'], 18, reuse=is_reuse, scope='mlp1' )
            net['mlp2'] = slim.fully_connected(
                net['mlp1'], 9, reuse=is_reuse, scope='mlp2' )
            net['logits'] = slim.fully_connected(
                net['mlp2'], numb_class, activation_fn=None, reuse=is_reuse, scope='logits')
            net['prediction'] = tf.nn.softmax( net['logits'] )
            return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net['input'] = input
        net['mlp1'] = slim.fully_connected(
            net['input'], 18, reuse=is_reuse, scope='mlp1')
        net['mlp2'] = slim.fully_connected(
            net['mlp1'], 9, reuse=is_reuse, scope='mlp2')
        net['logits'] = slim.fully_connected(
            net['mlp2'], numb_class, activation_fn=None, reuse=is_reuse, scope='logits')
        net['prediction'] = tf.nn.softmax(net['logits'])
        return net
'''
def network_architecture(input, numb_class, is_training, is_reuse, name=None):
    if not name == None:
        with tf.variable_scope(name):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net['input'] = input
            net['mlp1'] = slim.fully_connected(
                net['input'], 18, reuse=is_reuse, scope='mlp1' )
            net['mlp2'] = slim.fully_connected(
                net['mlp1'], 9, reuse=is_reuse, scope='mlp2')
            net['logits'] = slim.fully_connected(
                net['mlp2'], numb_class, activation_fn=None, reuse=is_reuse, scope='logits')
            net['prediction'] = tf.nn.softmax( net['logits'] )
            return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net['input'] = input
        net['mlp1'] = slim.fully_connected(
            net['input'], 18, reuse=is_reuse, scope='mlp1')
        net['mlp2'] = slim.fully_connected(
            net['mlp1'], 9, reuse=is_reuse, scope='mlp2')
        net['logits'] = slim.fully_connected(
            net['mlp2'], numb_class, activation_fn=None, reuse=is_reuse, scope='logits')
        net['prediction'] = tf.nn.softmax(net['logits'])
        return net

class PEMLPNET(object):
    def __init__(self, config, is_reuse, is_train, name=None):
        # Initialize setting
        if( is_train ):
            self.CLASS2WORD = config.CLASS2WORD
            self.n_class = config.numb_class
            self.batch_size = config.batch_size
            self.val_batch_size = config.val_batch_size
            self.max_epoch = config.epochs
            self.init_learning_rate = config.learning_rate
            self.decay_step = config.decay_step
            self.decay_factor = config.decay_factor
            self.val_freq = config.val_freq
            self.save_checkpoint_freq = config.save_checkpoint_freq
            self.network_name = name

            self.file_name = 'lr_{}_ds_{}'.format( self.init_learning_rate, config.dataset)
            self.exp_dir = op.join(config.exp_dir, self.file_name)
            self.save_checkpoint_dir = self.exp_dir
            self.log_dir = self.exp_dir
            self.is_training = config.is_training
            self.is_reuse = is_reuse

            # Setup session
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.allow_growth = True
            config_proto.allow_soft_placement = True
            self.sess = tf.Session(config=config_proto)

            # Create variables and placeholders
            self.pose = tf.placeholder(tf.float32, [None, config.input_dim])
            self.label = tf.placeholder(tf.int32, [None])

            ## define the network
            self.global_step = tf.Variable(0, trainable=False)
            self.net = self.load_model()
            self.val_loss = self.cross_entropy_loss(self.net['logits'], self.label)

            # if 0 <= self.decay_epoch < self.max_epoch:
            #     self.learning_rate = tf.train.exponential_decay(
            #         self.init_learning_rate,
            #         tf.maximum(0, self.global_step - self.decay_epoch),
            #         self.max_epoch - self.decay_epoch, 0.001)
            # else:
            #     self.learning_rate = self.init_learning_rate

            self.learning_rate = tf.train.exponential_decay(
                self.init_learning_rate,
                self.global_step,
                self.decay_step,
                self.decay_factor,
                staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # optimization
            self.train_var = tf.trainable_variables()
            #with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=self.is_reuse):
                self.train_op = self.optimizer.minimize(self.val_loss, var_list=self.train_var, global_step=self.global_step)

            self.checkpoint_saver = tf.train.Saver(max_to_keep=0)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.CLASS2WORD = config.CLASS2WORD
            self.n_class = config.numb_class
            self.init_learning_rate = config.learning_rate
            self.file_name = 'lr_{}_ds_{}'.format( self.init_learning_rate, config.dataset)
            self.checkpoint_test_dir = op.join(config.exp_dir, self.file_name)
            self.checkpoint_test_full_path = op.join( self.checkpoint_test_dir,config.checkpoint_test_name )
            self.is_training = config.is_training
            self.is_reuse = is_reuse
            self.network_name = name

            # Setup session
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.allow_growth = True
            config_proto.allow_soft_placement = True
            self.sess = tf.Session(config=config_proto)

            # Create variables and placeholders
            self.pose = tf.placeholder(tf.float32, [None, config.input_dim])

            ## define the network
            self.net = self.load_model()
            var_train = tf.trainable_variables()
            var_restore = [var for var in var_train if self.network_name in var.name]
            saver = tf.train.Saver(var_restore)
            saver.restore(self.sess, self.checkpoint_test_full_path)
        return

    def load_model(self):
        net = network_architecture(
                self.pose,
                self.n_class,
                self.is_training,
                self.is_reuse,
                self.network_name)
        return net

    def save_model(self, folder_save, step):
        if not op.exists( folder_save ):
            os.makedirs( folder_save )
        self.checkpoint_saver.save(self.sess, op.join(folder_save, 'checkpoint'), global_step=step)
        return

    def cross_entropy_loss(self, prediction, label ):
        if len(label.shape)==1:
            label = tf.one_hot(label, self.n_class)
        val_loss = tf.losses.softmax_cross_entropy(label, prediction, label_smoothing=0)

        return val_loss

    def train(self, pose_dataset ):
        print("%s #train# start training" % datetime.now())

        # tensorboard
        tflog_path = os.path.join(self.log_dir, 'log')
        if op.exists(tflog_path):
            shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)

        train_iter = 0
        for epoch in range(self.max_epoch):
            epoch_iter = int(pose_dataset.n_samples / self.batch_size)
            pose_dataset.finish_epoch()
            for i in range(epoch_iter):
                start_time = time.time()
                poses, labels = pose_dataset.next_batch(self.batch_size)
                feed_dict = {self.pose: poses,
                               self.label: labels}
                _, loss, b_prediction, b_label,step  = self.sess.run( [self.train_op, self.val_loss, self.net['prediction'], self.label, self.global_step], feed_dict=feed_dict)
                acc = 0
                b_correct_prediction = np.equal(np.argmax(b_prediction, 1), b_label)
                b_correct_prediction.astype("float32")
                b_correct_prediction = np.sum(b_correct_prediction)
                acc = 100*b_correct_prediction / self.batch_size
                train_iter += 1

                if( (train_iter%self.save_checkpoint_freq)==0 ):
                    self.save_model(op.join(self.save_checkpoint_dir, 'train'), step)


            print('Epoch: [%d/%d]: loss=%.06f, acc=%.02f%%\t\tTime-iter %.06fs'
                  % (epoch, self.max_epoch, loss, acc, time.time() - start_time))

        self.sess.close()

    def train_val(self, pose_dataset_train, pose_dataset_validation):
        print("%s #train# start training" % datetime.now())
        self.sess.run(tf.global_variables_initializer())

        train_iter = 0
        for epoch in range(self.max_epoch):
            epoch_iter = int(pose_dataset_train.n_samples / self.batch_size)
            pose_dataset_train.finish_epoch()
            for i in range(epoch_iter):
                start_time = time.time()
                poses, labels = pose_dataset_train.next_batch(self.batch_size)
                feed_dict = {self.pose: poses,
                             self.label: labels}
                _, loss, b_prediction, b_label, step = self.sess.run(
                    [self.train_op, self.val_loss, self.net['prediction'], self.label, self.global_step],
                    feed_dict=feed_dict)
                acc = 0
                b_correct_prediction = np.equal(np.argmax(b_prediction, 1), b_label)
                b_correct_prediction.astype("float32")
                b_correct_prediction = np.sum(b_correct_prediction)
                acc = b_correct_prediction / self.batch_size
                train_iter += 1

                if ((train_iter % self.save_checkpoint_freq) == 0):
                    self.save_model( op.join(self.save_checkpoint_dir, 'train_val'), step )

            print('Epoch[%d/%d]: loss=%.06f || acc=%.02f%% || (iter: %.06fs)'
                  % (epoch, self.max_epoch, loss, 100*acc, time.time() - start_time))
            if((epoch%self.val_freq)==0):
                acc_validation = self.validate(pose_dataset_validation)
                print('Validation accuracy @ epoch-%d: %.02f%%'
                      % (epoch, 100*acc_validation) )

        self.sess.close()

        return acc_validation

    def validate(self, pose_dataset):
        numb_batches = int(pose_dataset.n_samples / self.val_batch_size)
        acc_val = 0
        for i in range(numb_batches):
            b_poses, b_labels = pose_dataset.next_batch(self.val_batch_size)
            feed_dict = {self.pose: b_poses}
            b_prediction = self.sess.run(self.net['prediction'], feed_dict=feed_dict)
            b_correct_prediction = np.equal(np.argmax(b_prediction, 1), b_labels)
            b_correct_prediction.astype("float32")
            acc_val += np.sum(b_correct_prediction)
        acc_val = acc_val / float(pose_dataset.n_samples)

        return acc_val

    def test(self, pose):
        feed_dict = {self.pose: pose}
        score = self.sess.run(self.net['prediction'], feed_dict=feed_dict)
        predict_label = np.argmax(score)
        predict_emotion = self.CLASS2WORD[predict_label]
        return predict_emotion, np.squeeze(score), self.CLASS2WORD
