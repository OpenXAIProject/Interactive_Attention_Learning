from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import abc
import sys
from sklearn import linear_model, preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

TF_ENABLE_CONTROL_FLOW_V2=1
import os.path
import time
import IPython
import tensorflow as tf
import math

from models.GenericNeuralNet import GenericNeuralNet
from models.dataset import DataSet

import pdb

class IAL_NAP(GenericNeuralNet):
    def __init__(self,
                 num_features, 
                 steps, 
                 num_layers, 
                 hidden_units, 
                 embed_size,
                 alpha_np_shape,
                 np_shape,
                 np_batch,
                 weight_decay, 
                 variational_dropout,
                 n_batch_size,
                 num_test_points,
                 num_train_points,
                 **kwargs):

        self.num_features = num_features
        self.steps = steps
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.alpha_np_shape = alpha_np_shape
        self.np_shape = np_shape
        self.np_batch = np_batch
        self.embed_size = embed_size
        self.weight_decay = weight_decay
        self.variational_dropout = variational_dropout
        self.n_batch_size = n_batch_size
        self.num_test_points = num_test_points
        self.num_train_points = num_train_points
        super(IAL_NAP, self).__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        for portion in ['alpha_', 'beta_']:
            for var_name in ['mu_weights', 'mu_biases', 'sigma_weights', 'sigma_biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (portion, var_name))
                all_params.append(temp_tensor)
        for portion_ in ['output_weights', 'alpha_concat', 'beta_concat']:
            for var_name_ in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (portion_, var_name_))
                all_params.append(temp_tensor)       

        return all_params

    def get_attention_generator_params(self):
        params = []
        for i_ in range(len(tf.trainable_variables())):
            for variable_ in [
                              'alpha_/weights:0', \
                              'alpha_/mu_weights:0', \
                              'alpha_/sigma_weights:0', \
                              'alpha_/mu_biases:0', \
                              'alpha_/sigma_biases:0', \
                              'alpha_/alpha_NP_Rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',
                              'alpha_/alpha_NP_Rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',
                              'beta_/weights:0',
                              'beta_/mu_weights:0',
                              'beta_/sigma_weights:0',
                              'beta_/mu_biases:0',
                              'beta_/sigma_biases:0',
                              'beta_/beta_NP_Rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',
                              'beta_/beta_NP_Rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',
                              'alphaRnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',
                              'alphaRnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',
                              'betaRnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0',
                              'betaRnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0',
                              'alpha_concat/weights:0',
                              'alpha_concat/biases:0',
                              'beta_concat/weights:0',
                              'beta_concat/biases:0'
                              ]:

                if tf.trainable_variables()[i_].name == variable_:
                    params.append(tf.trainable_variables()[i_])
                else: continue

        return params

    def get_beta_generator_params(self):
        params = []
        for i_ in range(len(tf.trainable_variables())):
            for variable_ in [
                              'betaRnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0', \
                              'betaRnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0', \
                              'betaMU/weights:0', \
                              'betaMU/biases:0', \
                              'betaSIGMA/weights:0', \
                              'betaSIGMA/biases:0']:

                if tf.trainable_variables()[i_].name == variable_:
                    params.append(tf.trainable_variables()[i_])
                else: continue
        return params

    def get_without_cells_generator_params(self):
        params = []
        for i_ in range(len(tf.trainable_variables())):
            for variable_ in ['alphaMU/weights:0', \
                              'alphaMU/biases:0', \
                              'alphaSIGMA/weights:0', \
                              'alphaSIGMA/biases:0', 
                              'betaMU/weights:0', \
                              'betaMU/biases:0', \
                              'betaSIGMA/weights:0', \
                              'betaSIGMA/biases:0']:
                if tf.trainable_variables()[i_].name == variable_:
                    params.append(tf.trainable_variables()[i_])
                else: continue
        return params

    def retrain(self, num_steps, feed_dict):
        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])
        for step in range(num_steps):
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(shape=[None, self.steps, self.num_features], 
                                           dtype=tf.float32, 
                                           name='data'
                                          )
        labels_placeholder = tf.placeholder(shape=[None, 1], 
                                            dtype=tf.float32, 
                                            name='labels'
                                           )
        np_alpha_input_placeholder = tf.placeholder(shape=[None, self.steps, 1], 
                                           dtype=tf.float32, 
                                           name='alpha_annotation'
                                          )
        np_beta_input_placeholder = tf.placeholder(shape=[None, self.steps, self.num_features], 
                                           dtype=tf.float32, 
                                           name='beta_annotation'
                                          )

        return input_placeholder,\
               labels_placeholder,\
               np_alpha_input_placeholder,\
               np_beta_input_placeholder\

    def variable(name, shape, initializer):
        dtype = tf.float32
        var = tf.get_variable(
                              name, 
                              shape, 
                              initializer=initializer, 
                              dtype=dtype,
                              )
        return var

    def np_attention_op(self, str_id, rnn_outputs, hidden_units, embed_size, steps, context_vector, is_train):
        with tf.variable_scope(str_id+'_'):
            if str_id == 'alpha':
                initial_shape=[self.alpha_np_shape*1]
                p_att_shape = [self.alpha_np_shape, 1]
            elif str_id == 'beta':
                initial_shape=[self.np_shape*embed_size]
                p_att_shape = [self.np_shape, embed_size]
            else:
                raise ValueError('You must re-check the attention id. required to \'alpha\' or \'beta\'')

        #Create MU
        with tf.variable_scope(str_id+'_concat', reuse=tf.AUTO_REUSE):
            context_vector = tf.tile(context_vector, [tf.shape(rnn_outputs)[0], 1, 1])

            out_anno_concat = tf.concat([rnn_outputs, context_vector], axis=2)
            concat_w = tf.Variable(tf.random_normal(initial_shape, stddev=0.01), name='weights')
            concat_b = tf.get_variable(name='biases', shape=[p_att_shape[1]], initializer=tf.constant_initializer(0.0))
            concat_w = tf.reshape(concat_w, p_att_shape)

            #Linear Transformation
            output_concat =[]
            for _i in range(steps):
                tmp = tf.matmul(out_anno_concat[:, _i, :], concat_w) + concat_b
                output_concat.append(tmp)
            output = tf.reshape(tf.concat(output_concat, 1), [-1, steps, p_att_shape[1]])

        if str_id == 'alpha':
            squashed_att = tf.nn.softmax(output, 1)
            print('Done with generating alpha attention.')
        elif str_id == 'beta':
            squashed_att = tf.nn.tanh(output)
            print('Done with generating beta attention.')
        else:
            raise ValueError('You must re-check the attention id. required to \'Alpha\' or \'Beta\'')

        return squashed_att

    def inference(self, 
                  input_x, 
                  input_keep_probs, 
                  output_keep_probs, 
                  state_keep_probs, 
                  alpha_context_vector,
                  beta_context_vector, 
                  is_train
                  ):
        self.input_keep_probs_placeholder = tf.placeholder(tf.float32)
        self.output_keep_probs_placeholder = tf.placeholder(tf.float32)
        self.state_keep_probs_placeholder = tf.placeholder(tf.float32) 
        
        print ('Start building a model.')

        def single_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                                 dtype=tf.float32,
                                                 variational_recurrent=self.variational_dropout,
                                                 input_size=self.num_features,
                                                 input_keep_prob=input_keep_probs,
                                                 output_keep_prob=output_keep_probs,
                                                 state_keep_prob=state_keep_probs,
                                                )

        alpha_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
        beta_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

        with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
            self.W_emb = tf.get_variable('weights', shape=[self.num_features*self.embed_size], dtype=tf.float32)
            self.W_emb = tf.reshape(self.W_emb, shape=[self.num_features, self.embed_size])
            self.W_bias = tf.get_variable(name='biases', shape=[1], initializer=tf.constant_initializer(0.0)) 

        with tf.variable_scope('output_weights', reuse=tf.AUTO_REUSE):
            self.mu_weight = tf.get_variable('weights', shape=[self.embed_size*1])
            self.mu_weight = tf.reshape(self.mu_weight, shape=[self.embed_size, 1])
            self.mu_bias = tf.get_variable(name='biases', shape=[1], initializer=tf.constant_initializer(0.0)) 

        v_emb = []
        for _j in range(self.steps):
            embbed = tf.matmul(input_x[:, _j, :], self.W_emb) + self.W_bias
            v_emb.append(embbed)
        self.embedded_v = tf.reshape(tf.concat(v_emb, 1), [-1, self.steps, self.hidden_units])

        reversed_v_outputs = tf.reverse(self.embedded_v, [1])


        with tf.variable_scope("alphaRnn", reuse=tf.AUTO_REUSE) as scope:
            alpha_rnn_outputs, _ = tf.nn.dynamic_rnn(alpha_cell,
                                                     reversed_v_outputs,
                                                     dtype=tf.float32
                                                     )

        with tf.variable_scope("betaRnn", reuse=tf.AUTO_REUSE) as scope:
            beta_rnn_outputs, _ = tf.nn.dynamic_rnn(beta_cell,
                                                    reversed_v_outputs,
                                                    dtype=tf.float32
                                                    )
       
        #alpha
        alpha_embed_output = self.np_attention_op(
                                                   'alpha', 
                                                    alpha_rnn_outputs, 
                                                    self.hidden_units, 
                                                    self.embed_size, 
                                                    self.steps,
                                                    alpha_context_vector,
                                                    is_train,
                                                  )
        self.rev_alpha_embed_output = tf.reverse(alpha_embed_output, [1])

        #beta
        beta_embed_output = self.np_attention_op(
                                                 'beta', 
                                                  beta_rnn_outputs, 
                                                  self.hidden_units, 
                                                  self.embed_size, 
                                                  self.steps, 
                                                  beta_context_vector,
                                                  is_train
                                                 )
        self.rev_beta_embed_output = tf.reverse(beta_embed_output, [1])
        
        # attention_sum 
        c_i = tf.reduce_sum(self.rev_alpha_embed_output * (self.rev_beta_embed_output * self.embedded_v), 1)        
        
        logits = tf.matmul(c_i, self.mu_weight) + self.mu_bias
        return logits

    def predictions(self, logits):
        preds = tf.nn.sigmoid(logits, name='preds')
        return preds

    def mask_attention(self, masks, beta_embed_output):
        masked_attention_outputs=[]
        count_=0
        for z_ in range(masks.shape[0]):
            tmp=beta_embed_output[z_,:,:]*masks[count_, :, :]
            masked_attention_outputs.append(tmp)
            count_+=1
        masked_attention_outputs = tf.reshape(tf.concat(masked_attention_outputs, 1), [-1, self.steps, self.hidden_units])
        return masked_attention_outputs
