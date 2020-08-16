import numpy as np

class Anno(object):

    def __init__(self, x):

        x = x.astype(np.float32)

        self._x = x

        self._x_batch = np.copy(x)

        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x
   
    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            perm = np.arange(self._num_examples)
            self._x_batch = self._x_batch[perm, :, :]

            #Start next epoch
            start = 0 
            self._index_in_epoch = batch_size

        end=self._index_in_epoch
        return self._x[start:end]    