import tensorflow as tf
import numpy as np
import time
from six.moves import xrange
import scipy.io
import os
import collections
import random
import pdb

class NP(object):
    def __init__(self, steps, num_features, hidden_units, num_layers, embed_size, np_batch, batch_size):
        self._num_features = num_features
        self._hidden_units = hidden_units
        self._embed_size = embed_size
        self._steps = steps
        self._num_layers = num_layers
        self._np_batch = np_batch
        self._batch_size = batch_size

    def reparameterize(self, summary_r, mu_weight, mu_bias, sigma_weight, sigma_bias):
        mu = tf.matmul(summary_r, mu_weight) + mu_bias
        sigma = tf.nn.softplus(tf.matmul(summary_r, sigma_weight) + sigma_bias)
        return tf.distributions.Normal(loc=mu, scale=sigma), mu, sigma

    def subsampling(self, inp, np_batch):
        inp=tf.random.shuffle(inp)
        subsamples = inp[:np_batch, :, :]
        return subsamples

    def np_single_cell(self, hidden_units):
        lstm_cell = tf.contrib.rnn.LSTMCell(hidden_units)
        return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                             input_keep_prob=1,
                                             output_keep_prob=1,
                                             state_keep_prob=1,
                                             dtype=tf.float32,
                                             )

    def recurrent_model(self, inp, hiden_units, num_layers, str_id):
        with tf.variable_scope(str_id+'_NP_Rnn', reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.MultiRNNCell([self.np_single_cell(hiden_units) for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(
                                           cell,
                                           inp,
                                           dtype=tf.float32,
                                           )
            return outputs

    def inference(self, str_id, inp):
        if str_id == 'alpha':
            v_initial_shape = [1*1]
            v_p_att_shape   = [1,1]
            initial_shape   = [self._hidden_units*1]
            p_att_shape     = [self._hidden_units, 1]
        elif str_id == 'beta':
            initial_shape   = [self._hidden_units*self._embed_size]
            p_att_shape     = [self._hidden_units, self._embed_size]
        else:
            raise ValueError('You must re-check the attention id. required to \'Alpha\' or \'Beta\'')

        with tf.variable_scope(str_id+"_", reuse=tf.AUTO_REUSE):
            print('Build context vectors.')
            if str_id   == 'alpha':
                v_weight = tf.get_variable('weights', shape=v_initial_shape, dtype=tf.float32)
                v_weight = tf.reshape(v_weight, v_p_att_shape)
            elif str_id == 'beta':
                v_weight = tf.get_variable('weights', shape=initial_shape, dtype=tf.float32)
                v_weight = tf.reshape(v_weight, p_att_shape)
       
            mu_weight    = tf.get_variable('mu_weights', shape=initial_shape, dtype=tf.float32)
            sigma_weight = tf.get_variable('sigma_weights', shape=initial_shape, dtype=tf.float32)
            
            #Reshape
            mu_weight = tf.reshape(mu_weight, p_att_shape)
            sigma_weight = tf.reshape(sigma_weight, p_att_shape)

            mu_bias      = tf.get_variable('mu_biases', shape=[p_att_shape[1]], dtype=tf.float32)
            sigma_bias   = tf.get_variable('sigma_biases', shape=[p_att_shape[1]], dtype=tf.float32)

            subsamples = self.subsampling(inp, self._np_batch)

            v_emb = []
            for j_ in range(self._steps):
                embed = tf.matmul(subsamples[:, j_, :], v_weight)
                v_emb.append(embed)
            v_emb = tf.reshape(tf.concat(v_emb, 1), [-1, self._steps, p_att_shape[1]])

            r_i = self.recurrent_model(v_emb, self._hidden_units, self._num_layers, str_id)

            summary_r = tf.math.reduce_mean(r_i, axis=0)
            z_sample, mu, sigma = self.reparameterize(
                                                      summary_r, 
                                                      mu_weight, 
                                                      mu_bias, 
                                                      sigma_weight, 
                                                      sigma_bias
                                                     )
            z_sample=z_sample.sample([1])
            return z_sample