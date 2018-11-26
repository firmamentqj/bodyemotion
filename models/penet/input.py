import numpy as np
import math

class Dataset(object):
    def __init__(self, dataset):
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._index_in_epoch = 0
        self._epochs_complete = 0
        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        return

    def next_batch(self, batch_size):

        index_start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            if self._train:
                self._epochs_complete += 1
                index_start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                index_start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        index_end = self._index_in_epoch

        data, label = self._dataset.data(self._perm[index_start:index_end])
        return data, label


    @property
    def label(self):
        return self._dataset.get_labels()

    def finish_epoch(self):
        self._index_in_epoch = 0