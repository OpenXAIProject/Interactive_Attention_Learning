from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  
TF_ENABLE_CONTROL_FLOW_V2=1

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
from sklearn.metrics import roc_curve, auc

import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.learn.python.learn.datasets import base

from models.hessians import hessian_vector_product
from models.dataset import DataSet
from models.anno import Anno
from models.py_utils import cprint
import pdb
from models.metric import *
from models.load_data import Baseline_version_load_Data
import collections
from models.np import NP


class GenericNeuralNet(object):
    def __init__(self, **kwargs):
        np.random.seed(0)
        tf.set_random_seed(0)

        self.batch_size = kwargs.pop('batch_size')
        self.data_sets  = kwargs.pop('data_sets')
        self.anno       = kwargs.pop('anno')

        log_dir = kwargs.pop('log_dir', 'log')
        self.model_name = kwargs.pop('model_name')
        self.retrain_model_name = kwargs.pop('retrain_model_name')
        #self.num_classes = kwargs.pop('num_classes')
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')        
        self.decay_epochs = kwargs.pop('decay_epochs')
        self.num_sampling = kwargs.pop('num_sampling')
        self.train_dir    = kwargs.pop('train_dir')
        self.retrain_dir  = kwargs.pop('retrain_dir')


        if 'input_keep_probs' and 'output_keep_probs' and 'state_keep_probs' in kwargs:
            self.input_keep_probs = kwargs.pop('input_keep_probs')
            self.output_keep_probs = kwargs.pop('output_keep_probs')
            self.state_keep_probs = kwargs.pop('state_keep_probs')
        else:
            self.input_keep_probs = None
            self.output_keep_probs = None
            self.state_keep_probs = None

        if 'mini_batch' in kwargs: 
            self.mini_batch = kwargs.pop('mini_batch')        
        else: self.mini_batch = True

        if 'damping' in kwargs:
            self.damping = kwargs.pop('damping')

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        #Initialize_session
        print('Initialize a sesion.')
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

        #Setup input
        print('Setup Input')
        self.input_placeholder, self.labels_placeholder, self.alpha_np_input_placeholder, self.beta_np_input_placeholder  = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]

        self.np = NP(
                     self.steps,
                     self.num_features,
                     self.hidden_units,
                     self.num_layers,
                     self.embed_size,
                     self.np_batch,
                     self.n_batch_size,
                    )

        #Build context vector
        self.alpha_context_vector = self.np.inference("alpha", self.alpha_np_input_placeholder)
        self.beta_context_vector = self.np.inference("beta", self.beta_np_input_placeholder)

        self.logits = self.inference(self.input_placeholder,
                                     self.input_keep_probs,
                                     self.output_keep_probs,
                                     self.state_keep_probs,
                                     self.alpha_context_vector,
                                     self.beta_context_vector,
                                     is_train="training",
                                     )

        self.preds = self.predictions(self.logits)
        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
                                                                              self.logits, 
                                                                              self.labels_placeholder
                                                                              )

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(
                                         self.initial_learning_rate, 
                                         name='learning_rate', 
                                         trainable=False
                                         )

        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.update_learning_rate_op = tf.assign(
                                                 self.learning_rate, 
                                                 self.learning_rate_placeholder
                                                 )

        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        self.retrain_op = self.get_retrain_op(self.total_loss, self.global_step, self.learning_rate)



        self.auc_op, self.update_op = self.get_auc_op(self.preds, 
                                      self.labels_placeholder
                                      )

        #Setup misc
        self.saver = tf.train.Saver(max_to_keep=30)

        
        #Setup gradients and Hessians
        self.params = self.get_all_params()

        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_total_loss_op = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(self.params, 
            self.grad_total_loss_op)]

        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        self.grad_loss_no_reg_op = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(self.params, 
            self.grad_loss_no_reg_op)]

        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)
        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)

        self.influence_op = tf.add_n([tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) \
                                      for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)
        
        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)
        self.retrain_checkpoint_file = os.path.join(self.retrain_dir, "%s-checkpoint" % self.retrain_model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train, self.anno, dropout=True)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test, self.anno, dropout=True)

        init = tf.global_variables_initializer()        
        self.sess.run(init)
        self.vec_to_list = self.get_vec_to_list_fn()

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))
        print('Total number of parameters: %s' % self.num_params)

        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p_ in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p_)])
                cur_pos += len(p_)
            #assert cur_pos == len(v)
            return return_list
        return vec_to_list

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove, dropout=False):
        num_examples = data_set.x.shape[0]
        idx = np.array([True]*num_examples, dtype=bool)
        idx[idx_to_remove] = False
        if dropout==True:
            feed_dict = {
                         self.input_placeholder:data_set.x[idx, :, :],
                         self.labels_placeholder: data_set.labels[idx],
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,
                        }


        elif dropout==False:       
            feed_dict = {
                         self.input_placeholder: data_set.x[idx, :, :],
                         self.labels_placeholder: data_set.labels[idx],
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }
        return feed_dict

    def fill_feed_dict_with_all_but_some_ex(self, data_set, indices_to_remove, dropout=False):
        num_examples = data_set.x.shape[0]
        idx = np.array([True]*num_examples, dtype=bool)
        idx[indices_to_remove] = False
        if dropout==True:
            feed_dict = {
                         self.input_placeholder: data_set.x[idx, :, :],
                         self.labels_placeholder: data_set.labels[idx],
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,
                        }

        elif dropout==False:
            feed_dict = {
                         self.input_placeholder: data_set.x[idx, :, :],
                         self.labels_placeholder: data_set.labels[idx],
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }      
        return feed_dict


    def reset_datasets(self):
        self.data_sets.train.reset_batch()
        self.data_sets.test.reset_batch()


    def minibatch_mean_eval(self, ops, data_set):
        num_examples = data_set.num_examples
        num_iter = int(num_examples / self.batch_size)

        self.reset_datasets()

        ret = []
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set, dropout=False)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)
            if len(ret)==0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])

            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))
        return ret


    def retrain(self, num_steps, feed_dict):        
        for step in xrange(num_steps):   
            self.sess.run(self.train_op, feed_dict=feed_dict)


    def update_learning_rate(self, step):
        num_steps_in_epoch = self.num_train_examples / self.batch_size
        epoch = step // num_steps_in_epoch

        multiplier = 1
        if epoch < self.decay_epochs[0]:
            multiplier = 1
        elif epoch < self.decay_epochs[1]:
            multiplier = 0.1
        else:
            multiplier = 0.01
        
        self.sess.run(
                      self.update_learning_rate_op, 
                      feed_dict={self.learning_rate_placeholder: multiplier * self.initial_learning_rate}
                     )

    def load_checkpoint(self, iter_to_load, do_checks=False):
        checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load) 
        self.saver.restore(self.sess, checkpoint_to_load)
        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()


    def retrain_load_checkpoint(self, iter_to_load, do_checks=True):
        print('retrain_dir is open.')
        print('retrain dir: %s' %(self.retrain_dir))
        checkpoint_to_load = "%s-%s" % (self.retrain_checkpoint_file, iter_to_load) 
        self.saver.restore(self.sess, checkpoint_to_load)
        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()

    def regularization(self):
        all_variables=tf.trainable_variables()
        l2_losses=[]
        for variable in all_variables:
            variable = tf.cast(variable, tf.float32)
            l2_losses.append(tf.nn.l2_loss(variable))
        regul = 0.004*tf.reduce_sum(l2_losses)
        return regul

    def attention_generator_regularization(self):
        all_variables=self.get_attention_generator_params()
        l2_losses=[]
        for variable in all_variables:
            variable = tf.cast(variable, tf.float32)
            l2_losses.append(tf.nn.l2_loss(variable))
        regul = 0.004*tf.reduce_sum(l2_losses)
        return regul

    def get_retrain_op(self, total_loss, global_step, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        att_generator_params = self.get_attention_generator_params()
        attention_regul = self.attention_generator_regularization()
        retrain_op = optimizer.minimize(total_loss+attention_regul, var_list=att_generator_params, global_step=global_step)
        return retrain_op

    def get_train_op(self, total_loss, global_step, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        regul = self.regularization()
        train_op = optimizer.minimize(total_loss+regul, global_step=global_step)
        return train_op

    def get_train_sgd_op(self, total_loss, global_step, learning_rate=0.001):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        regul = self.regularization()
        train_op = optimizer.minimize(total_loss+regul, global_step=global_step)
        return train_op

    def accuracy(self, preds, labels):
        return (100.0 * np.sum(preds == labels)
                / preds.shape[0])

    def RMSE(self, p, y): 
        N = p.shape[0]
        diff = p - y 
        return np.sqrt((diff**2).mean())

    def get_auc_op(self, predictions, labels): 
        return tf.metrics.auc(labels, predictions)

    def loss(self, logits, labels):
        prob = tf.nn.sigmoid(logits)
        cross_entropy = - tf.reduce_sum(labels * tf.log(prob + 1e-15) + (1 - labels) * tf.log(1 - prob + 1e-15), axis=1)
        #cross_entropy = - tf.reduce_sum(labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits), axis=1)

        indiv_loss_no_reg = cross_entropy
        loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return total_loss, loss_no_reg, indiv_loss_no_reg

    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block
        return feed_dict

    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)

    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v) 
        fmin_grad_fn = self.get_fmin_grad_fn(v) 
        cg_callback = self.get_cg_callback(v, verbose=True)
        fmin_results = fmin_ncg(
                                f=fmin_loss_fn, 
                                x0=np.concatenate(v), 
                                fprime=fmin_grad_fn, 
                                fhess_p=self.get_fmin_hvp, 
                                callback=cg_callback, 
                                avextol=1e-8, 
                                maxiter=100,
                               )
        return self.vec_to_list(fmin_results)

    def get_fmin_loss_fn(self, v):
        def get_fmin_loss(x):
            print('get_fmin_loss.')
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            print('get_fmin_grad.')
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return get_fmin_grad

    def get_fmin_hvp(self, x, p):
        print('get_fmin_hvp')
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))
        return np.concatenate(hessian_vector_val)

    def get_cg_callback(self, v, verbose=True):
        print('get_cg_callback.')
        fmin_loss_fn = self.get_fmin_loss_fn(v)

        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            return 0.5*np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)
 
        def cg_callback(x):
            #x is current params
            v = self.vec_to_list(x)
            idx_to_remove=np.array(self.data_sets.train.labels.shape[0])
            for i_ in range(idx_to_remove):
                single_train_feed_dict=self.fill_feed_dict_with_one_ex(self.data_set.train, idx_to_remove[i_], dropout=False)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                guad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (guad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))
            return cg_callback

    def minibatch_hessian_vector_val(self, v):
        cprint('minibatch_hessian_vector_val', bg_color='y')
        num_examples = self.num_train_examples
        if self.mini_batch  == True:
            batch_size = 20
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)
        self.reset_datasets()
        hessian_vector_val = None
        for i in range(1000):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, self.anno, batch_size=batch_size, dropout=False)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]
        hessian_vector_val = [a + self.damping * b for (a,b) in zip(hessian_vector_val, v)]
        return hessian_vector_val


    # The MIT License Copyright (c) 2017 Pang Wei Koh and Percy Liang
    # ===================================================================================
    def get_influence_on_test_loss(self, 
                                   test_indices, 
                                   train_idx, 
                                   approx_type='cg', 
                                   approx_params=None, 
                                   force_refresh=True, 
                                   test_description=None,
                                   loss_type='normal_loss',
                                   X=None, 
                                   Y=None
                                   ):

        if train_idx is None: 
            if (X is None) or (Y is None): 
                raise ValueError('X and Y must be specified if using phantom points.')

            if X.shape[0] != len(Y): 
                raise ValueError('X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None): 
                raise ValueError('X and Y cannot be specified if train_idx is specified.')

        #Test Loss
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)
        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()
        if test_description is None:
            test_description = "training_indices"

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test_jay-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            print("force_refresh: False")
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                                               test_grad_loss_no_reg_val,
                                               approx_type,
                                               approx_params
                                               )

            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)
        
        start_time = time.time()

        num_to_remove = len(train_idx)
        predicted_loss_diffs = np.zeros([num_to_remove])
        for counter, idx_to_remove in enumerate(train_idx):            
            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove, self.anno,  dropout=False)      
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples

        predicted_loss_diff_filename = os.path.join(self.train_dir, '%s-%s-%s-predicted_loss_diffs.npz' % (self.model_name, approx_type, loss_type))
        np.savez(predicted_loss_diff_filename, predicted_loss_diffs=predicted_loss_diffs)
        
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))
        print('Number of predicted_loss_diffs before collecting: %s' % predicted_loss_diffs.shape[0])
        return predicted_loss_diffs


    # The MIT License Copyright (c) 2017 Pang Wei Koh and Percy Liang
    # ===================================================================================
    def get_feature_influence_on_test_loss(self, 
                                   test_indices, 
                                   train_idx, 
                                   perturbed_input,
                                   perturbed_labels,
                                   approx_type='cg', 
                                   approx_params=None, 
                                   force_refresh=True, 
                                   test_description=None,
                                   loss_type='normal_loss',
                                   X=None, 
                                   Y=None
                                   ):
        if train_idx is None: 
            if (X is None) or (Y is None): 
                raise ValueError('X and Y must be specified if using phantom points.')

            if X.shape[0] != len(Y): 
                raise ValueError('X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None): 
                raise ValueError('X and Y cannot be specified if train_idx is specified.')

        #Test Loss
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)
        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()
        if test_description is None:
            test_description = "jay_training_indices"

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test_jay-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            print("force_refresh: False")
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                                               test_grad_loss_no_reg_val,
                                               approx_type,
                                               approx_params
                                               )

            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)
        
        start_time = time.time()

        num_to_remove = len(perturbed_input)
        predicted_loss_diffs = np.zeros([num_to_remove])
        for counter, idx_to_remove in enumerate(np.arange(perturbed_input.shape[0])):            
            single_train_feed_dict = self.feature_fill_feed_dict_with_one_ex(perturbed_input, perturbed_labels, idx_to_remove, self.anno,  dropout=False)      
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples

        predicted_loss_diff_filename = os.path.join(self.train_dir, 'feature_level-%s-%s-%s-predicted_loss_diffs.npz' % (self.model_name, approx_type, loss_type))
        np.savez(predicted_loss_diff_filename, predicted_loss_diffs=predicted_loss_diffs)
        
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))
        print('Number of predicted_loss_diffs before collecting: %s' % predicted_loss_diffs.shape[0])
        return predicted_loss_diffs


    def feature_fill_feed_dict_with_one_ex(self, data_set_x, data_set_labels, target_idx, anno, dropout=False):
        input_feed = data_set_x[target_idx, :, :].reshape(-1, data_set_x.shape[1], data_set_x.shape[2])
        labels_feed = data_set_labels[target_idx].reshape(-1, data_set_labels.shape[1])

        alpha = anno.alpha.next_batch(self.batch_size)
        beta = anno.beta.next_batch(self.batch_size)
        
        if dropout==True:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,
                        }

        elif dropout==False:       
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }                        
        return feed_dict


    def get_ut_from_trainingpoints(self, num_to_choose, dropout=True):
        train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train, self.anno, dropout=True)
        total_preds=[]        
        for i_ in range(self.num_sampling):
            preds = self.sess.run([self.preds], feed_dict=train_feed_dict)
            total_preds.append(preds)
        
        preds = np.mean(total_preds, axis=0).reshape(-1, 1)
        ut_preds = np.var(total_preds, axis=0).reshape(-1, 1)

        chosen_training_indices = np.concatenate(np.argsort(ut_preds, axis=0)[-num_to_choose:], -1)
        return chosen_training_indices


    def get_feature_importance_with_counterfactual(self,
                                                   inputs,
                                                   labels,
                                                   masked_inputs,
                                                   masked_labels,
                                                   ):
        start_time = time.time()
        idv_feature_importance = np.zeros([len(inputs)])
        for counter, idx_to_remove in enumerate(np.arange(inputs.shape[0])):
            original_single_train_feed_dict = self.feature_fill_feed_dict_with_one_ex(inputs, labels, idx_to_remove, self.anno,  dropout=False)
            original_pred = self.sess.run(self.preds, feed_dict=original_single_train_feed_dict)
            masked_single_train_feed_dict = self.feature_fill_feed_dict_with_one_ex(masked_inputs, masked_labels, idx_to_remove, self.anno,  dropout=False)
            masked_pred = self.sess.run(self.preds, feed_dict=masked_single_train_feed_dict)
            idv_feature_importance[counter] = np.abs(original_pred - masked_pred)

        pred_diff_filename = os.path.join(self.train_dir, 'counterfactual_feature_level-%s-predicted_loss_diffs.npz' % (self.model_name))
        np.savez(pred_diff_filename, idv_feature_importance=idv_feature_importance)

        duration = time.time() - start_time
        return idv_feature_importance


    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100, loss_type='normal_loss'):
        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        else:
            raise ValueError('Loss must be specified')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))
            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end], self.anno, dropout=False)
                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]
        
        return test_grad_loss_no_reg_val


    # The MIT License Copyright (c) 2017 Pang Wei Koh and Percy Liang
    # ===================================================================================
    def get_grad_of_influence_wrt_input(self, 
                                        train_indices, 
                                        test_indices, 
                                        approx_type='cg', 
                                        approx_params=None, 
                                        force_refresh=True, 
                                        verbose=True, 
                                        test_description=None,
                                        loss_type='normal_loss'
                                        ):

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)            

        if verbose: 
            print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
        
        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: 
                print('Loaded inverse HVP from %s' % approx_filename)
        else:            
            inverse_hvp = self.get_inverse_hvp(
                                               test_grad_loss_no_reg_val,
                                               approx_type,
                                               approx_params,
                                               verbose=verbose
                                               )

            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: 
                print('Saved inverse HVP to %s' % approx_filename)            
        
        duration = time.time() - start_time
        if verbose: 
            print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                                                                       self.data_sets.train,  
                                                                       train_idx,
                                                                       self.anno,
                                                                       dropout=False,
                                                                      )

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]            
            
            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val


    def fill_feed_dict_with_batch(self, data_set, anno, batch_size=0, dropout=False):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)
        alpha = anno.alpha.next_batch(self.batch_size)
        beta = anno.beta.next_batch(self.batch_size)

        if dropout==True:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,            
                        }

        elif dropout==False:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }       
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices, anno, dropout=False):
        input_feed = data_set.x[target_indices, :, :].reshape(-1, data_set.x.shape[1], data_set.x.shape[2])
        labels_feed = data_set.labels[target_indices, :].reshape(-1, data_set.labels.shape[1])
        
        alpha = anno.alpha.next_batch(self.batch_size)
        beta = anno.beta.next_batch(self.batch_size)

        if dropout==True:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,
                        }

        elif dropout==False:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx, anno, dropout=False):
        input_feed = data_set.x[target_idx, :, :].reshape(-1, data_set.x.shape[1], data_set.x.shape[2])
        labels_feed = data_set.labels[target_idx].reshape(-1, data_set.labels.shape[1])

        alpha = anno.alpha.next_batch(self.batch_size)
        beta = anno.beta.next_batch(self.batch_size)
        
        if dropout==True:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,
                        }

        elif dropout==False:       
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }                        
        return feed_dict



    def train(self,
              num_steps,
              save_checkpoints=True,
              verbose=True,
              dropout=True
              ):

        if verbose:
            print('Training for %s steps' % num_steps)

        sess = self.sess

        for step in range(num_steps):
            self.update_learning_rate(step)
            start_time=time.time()
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, self.anno, dropout=dropout)
            _, loss_val, loss_no_reg, indiv_loss_no_reg, logits, preds = sess.run([self.train_op, 
                                                                                   self.total_loss, 
                                                                                   self.loss_no_reg, 
                                                                                   self.indiv_loss_no_reg, 
                                                                                   self.logits,
                                                                                   self.preds], 
                                                                                   feed_dict=feed_dict
                                                                                   )

            duration = time.time() - start_time

            if verbose:
                if step % 100 == 0:
                    cprint('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration), bg_color='b')
            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:    
                if save_checkpoints: 
                    self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose: 
                    self.print_model_eval()
                self.Evaluation()

            

    def fill_feed_dict_with_all_ex(self, data_set, anno, dropout=False):
        alpha = anno.alpha.next_batch(self.batch_size)
        beta = anno.beta.next_batch(self.batch_size)

        if dropout==True:
            feed_dict = {
                         self.input_placeholder: data_set.x,
                         self.labels_placeholder: data_set.labels,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,
                        }

        elif dropout==False:      
            feed_dict = {
                         self.input_placeholder: data_set.x,
                         self.labels_placeholder: data_set.labels,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,                       
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }
        return feed_dict


    def print_model_eval(self,):
        params_val = self.sess.run(self.params)
        self.sess.run(tf.local_variables_initializer())

        train_total_loss=[]
        train_total_preds=[]
        for j_ in range(self.num_sampling):
            train_loss, train_preds = self.sess.run([
                                                     self.total_loss,
                                                     self.preds],
                                                     feed_dict=self.all_train_feed_dict
                                                    )

            train_total_loss.append(train_loss)
            train_total_preds.append(train_preds)

        train_loss = np.mean(train_total_loss, axis=0)
        train_preds = np.mean(train_total_preds, axis=0)

        train_roc, averaged_train_auc = ROC_AUC(train_preds, self.data_sets.train.labels)
        
        eval_total_loss=[]
        eval_total_preds=[]
        for j_ in range(self.num_sampling):
            test_loss, test_preds = self.sess.run(
                                                  [self.total_loss,
                                                   self.preds,],
                                                   feed_dict=self.all_test_feed_dict
                                                 )
            eval_total_loss.append(test_loss)
            eval_total_preds.append(test_preds)

        eval_loss = np.mean(eval_total_loss, axis=0)
        total_preds = np.mean(eval_total_preds, axis=0)

        eval_roc, averaged_test_auc = ROC_AUC(total_preds, self.data_sets.test.labels)

        cprint('Train loss (w/o reg) on all data: %s' % train_loss, bg_color='o')
        cprint('Test loss (w/o reg) on all data: %s' % test_loss, bg_color='o')
        cprint('Train AUC:  %s' % averaged_train_auc, bg_color='o')
        cprint('Test AUC:   %s' % averaged_test_auc, bg_color='o')


    def Evaluation(self):
        
        test_feed_dict = self.fill_feed_dict_with_all_ex(
                                                         self.data_sets.final_evaluation, 
                                                         self.anno, 
                                                         dropout=True,
                                                         )

        print("[Inputs]Number of evaluation instance: %s" % test_feed_dict[self.input_placeholder].shape[0])
        print("[Labels]Number of evaluation instance: %s" % test_feed_dict[self.labels_placeholder].shape[0])

        eval_total_loss=[]
        eval_total_preds=[]
        for j_ in range(self.num_sampling):
            loss, preds = self.sess.run(
                                        [self.total_loss, 
                                         self.preds], 
                                         feed_dict=test_feed_dict,
                                        )
            eval_total_loss.append(loss)
            eval_total_preds.append(preds)

        averaged_eval_loss = np.mean(eval_total_loss, axis=0)
        total_preds = np.mean(eval_total_preds, axis=0)

        eval_roc, eval_auc = ROC_AUC(total_preds, test_feed_dict[self.labels_placeholder])
        cprint("[*] Evaluation loss: %.8f, Evaluation AUC:  %.8f" % (averaged_eval_loss, eval_auc), bg_color='r')


    def Feature_contributions_for_chosen_instance(self, chosen_indices, type_instance=None):
        if type_instance == 'train':
            print("Compute feature contribution for train instances.")
            input_feed = self.data_sets.train.x[chosen_indices, :, :].reshape(-1, self.data_sets.train.x.shape[1], self.data_sets.train.x.shape[2])
            labels_feed = self.data_sets.train.labels[chosen_indices, :].reshape(-1, self.data_sets.train.labels.shape[1])

            W_emb, W_out = self.sess.run(
                                         [self.W_emb, 
                                          self.mu_weight,]
                                        )

            feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.train, chosen_indices, self.anno, dropout=True)
            alpha, beta = self.sess.run(
                                        [self.rev_alpha_embed_output, 
                                         self.rev_beta_embed_output,],
                                         feed_dict=feed_dict,
                                        )
            print("Number of chosen training instances: %s" % len(chosen_indices))

        elif type_instance == 'test':
            print("Compute feature contribution for test instances.")
            input_feed = self.data_sets.test.x[chosen_indices, :, :].reshape(-1, self.data_sets.test.x.shape[1], self.data_sets.test.x.shape[2])
            labels_feed = self.data_sets.test.labels[chosen_indices, :].reshape(-1, self.data_sets.test.labels.shape[1])

            W_emb, W_out = self.sess.run(
                                         [self.W_emb, self.mu_weight,]
                                        )

            feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, chosen_indices, self.anno, dropout=True)
            alpha, beta = self.sess.run(
                                        [self.rev_alpha_embed_output, 
                                         self.rev_beta_embed_output,],
                                         feed_dict=feed_dict,
                                        )
            print("Number of chosen test instances: %s" % len(chosen_indices))

        elif type_instance == 'final_evaluation':
            print("Compute feature contribution for final_evaluation instances.")
            input_feed = self.data_sets.final_evaluation.x[chosen_indices, :, :].reshape(-1, self.data_sets.final_evaluation.x.shape[1], self.data_sets.final_evaluation.x.shape[2])
            labels_feed = self.data_sets.final_evaluation.labels[chosen_indices, :].reshape(-1, self.data_sets.final_evaluation.labels.shape[1])

            W_emb, W_out = self.sess.run(
                                         [self.W_emb, self.mu_weight,]
                                        )

            feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.final_evaluation, chosen_indices, self.anno, dropout=True)
            alpha, beta = self.sess.run(
                                        [self.rev_alpha_embed_output,
                                         self.rev_beta_embed_output,],
                                         feed_dict=feed_dict,
                                        )
            print("Number of chosen final_evaluation instances: %s" % len(chosen_indices))

        total_interpret = []
        for j_ in range(self.steps):
            timestep_interpret = []
            for k_ in range(self.num_features):
                elemwise = beta[:, j_, :] * W_emb[k_, :]
                interpret_value = np.sum(alpha[:, j_], 1) * np.sum(np.matmul(elemwise, W_out), 1) * input_feed[:, j_, k_]
                timestep_interpret.append(interpret_value)
            total_interpret.append(timestep_interpret)

        chosen_instances_interpretation = np.array(total_interpret)    
        
        origin = np.zeros((chosen_instances_interpretation.shape[2], chosen_instances_interpretation.shape[0], chosen_instances_interpretation.shape[1]))
        for ins_ in range(chosen_instances_interpretation.shape[2]):
            origin[ins_, :, :] = chosen_instances_interpretation[:, :, ins_]

        return origin, input_feed, labels_feed


    def Collect_incorrect_test_indices(self, num_to_choose):
        test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test, self.anno, dropout=True)

        total_preds=[]        
        for i_ in range(self.num_sampling):
            preds = self.sess.run([self.preds], feed_dict=test_feed_dict)
            total_preds.append(preds)
        preds = np.mean(total_preds, axis=0).reshape(-1, 1)

        tmp=[]
        for i_ in range(self.data_sets.test.labels.shape[0]):
            diff = self.data_sets.test.labels[i_] - preds[i_]
            tmp.append(diff)
        test_indices = np.concatenate(np.argsort(np.abs(tmp), axis=0)[-num_to_choose:], -1)
        return test_indices

 
    def Sampling_prediction_with_chosen_test_indices(self, target_indices, dropout=True, num_sampling=50, num_to_choose=50):
        test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, target_indices, self.anno, dropout=dropout)

        input_feed = self.data_sets.test.x[target_indices, :, :].reshape(-1, self.data_sets.test.x.shape[1], self.data_sets.test.x.shape[2])
        labels_feed = self.data_sets.test.labels[target_indices, :].reshape(-1, self.data_sets.test.labels.shape[1])
        
        all_preds=np.zeros((num_sampling, len(target_indices), 1))
        for i_ in range(num_sampling):
            preds = self.sess.run(self.preds, feed_dict=test_feed_dict)
            all_preds[i_, :, :] = preds

        mean_preds = np.mean(all_preds, axis=0)
        UT_preds = np.var(all_preds, axis=0)

        chosen_indices = np.concatenate(np.argsort(UT_preds, axis=0)[-num_to_choose:], -1)

        chosen_mean_preds = mean_preds[chosen_indices]
        chosen_UT_preds = UT_preds[chosen_indices]
        chosen_input_feed = input_feed[chosen_indices]
        chosen_labels_feed = labels_feed[chosen_indices]
        return chosen_indices, chosen_mean_preds, chosen_UT_preds, chosen_input_feed, chosen_labels_feed


    def UT_Feature_contributions_for_chosen_instance(self, first_chosen_indices, second_chosen_indices, type_instance=None):
        if type_instance == 'test':
            print("Compute feature contribution for test instances.")
            input_feed = self.data_sets.test.x[first_chosen_indices[second_chosen_indices], :, :].reshape(-1, self.data_sets.test.x.shape[1], self.data_sets.test.x.shape[2])
            labels_feed = self.data_sets.test.labels[first_chosen_indices[second_chosen_indices], :].reshape(-1, self.data_sets.test.labels.shape[1])

            W_emb, W_out = self.sess.run(
                                         [self.W_emb, self.mu_weight,]
                                        )

            feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, first_chosen_indices[second_chosen_indices], self.anno, dropout=True)
            alpha, beta = self.sess.run(
                                        [self.rev_alpha_embed_output, 
                                         self.rev_beta_embed_output,],
                                         feed_dict=feed_dict
                                        )
            print("Number of chosen test instances: %s" % len(first_chosen_indices[second_chosen_indices]))
        else:
            ValueError("Need to specify type_instance.")

        total_interpret = []
        for j_ in range(self.steps):
            timestep_interpret = []
            for k_ in range(self.num_features):
                elemwise = beta[:, j_, :] * W_emb[k_, :]
                interpret_value = np.sum(alpha[:, j_], 1) * np.sum(np.matmul(elemwise, W_out), 1) * input_feed[:, j_, k_]
                timestep_interpret.append(interpret_value)
            total_interpret.append(timestep_interpret)

        chosen_instances_interpretation = np.array(total_interpret)    
        
        origin = np.zeros((chosen_instances_interpretation.shape[2], chosen_instances_interpretation.shape[0], chosen_instances_interpretation.shape[1]))
        for ins_ in range(chosen_instances_interpretation.shape[2]):
            origin[ins_, :, :] = chosen_instances_interpretation[:, :, ins_]
        return origin, input_feed, labels_feed


    def train_masked_fill_feed_dict(self, anno, input_feed, labels_feed, dropout=False):
        input_feed = input_feed.reshape(-1, self.data_sets.train.x.shape[1], self.data_sets.train.x.shape[2])
        labels_feed = labels_feed.reshape(-1, self.data_sets.train.labels.shape[1])
        
        alpha = anno.alpha.next_batch(self.batch_size)
        beta = anno.beta.next_batch(self.batch_size)

        if dropout==True:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,    
                         self.input_keep_probs_placeholder: self.input_keep_probs,
                         self.output_keep_probs_placeholder: self.output_keep_probs,
                         self.state_keep_probs_placeholder: self.state_keep_probs,            
                        }

        elif dropout==False:
            feed_dict = {
                         self.input_placeholder: input_feed,
                         self.labels_placeholder: labels_feed,
                         self.alpha_np_input_placeholder: alpha, 
                         self.beta_np_input_placeholder: beta,    
                         self.input_keep_probs_placeholder: 1,
                         self.output_keep_probs_placeholder: 1,
                         self.state_keep_probs_placeholder: 1,
                        }       
        return feed_dict

    def retrain_train_instance_after_oracle(self,
                                           iter_to_load,
                                           num_steps,
                                           anno,
                                           retrain_dir,
                                           anno_dataset,
                                           save_checkpoints=True,
                                           verbose=True,
                                           dropout=True,
                                           do_checks=True,
                                           ):
        if not os.path.exists(self.retrain_dir):
            os.makedirs(retrain_dir) 

        if verbose: print('Retraining for %s steps' % num_steps)

        # Load optimized checkpoints
        self.load_checkpoint(iter_to_load)
        sess = self.sess

        logfile = open(os.path.join(self.retrain_dir, 'log.txt'), 'wt')
        total_duration=[]
        for step in range(num_steps):
            self.update_learning_rate(step)
            start_time=time.time()
            line = ""
            feed_dict = self.train_masked_fill_feed_dict(
                                                         anno=anno, 
                                                         input_feed=anno_dataset.retrain.x, 
                                                         labels_feed=anno_dataset.retrain.labels, 
                                                         dropout=dropout
                                                        )

            _ = sess.run(
                         self.retrain_op,   
                         #tf.no_op(),
                         feed_dict=feed_dict,
                        )

            duration = time.time() - start_time
            total_duration.append(duration)

            if verbose:
                if step % 1 == 0:
                    loss_val, = sess.run(
                                         [self.total_loss,], 
                                          feed_dict=feed_dict
                                        )
                    cprint('Step %d: loss = %.8f' % (step, loss_val), bg_color='b')
                    line += "Step %d: loss = %.8f\n" % (step, loss_val)
            if (step + 1) % 1 == 0 or (step + 1) == num_steps:
                if save_checkpoints: self.saver.save(sess, self.retrain_checkpoint_file, global_step=step)
            if do_checks: 
                averaged_eval_loss, eval_auc = self.print_evaluation_after_retraining()
                print('Total duration: %s'   % (sum(total_duration)))
                line += "Total duration: %s\n" % (sum(total_duration))
                line += "[*] Evaluation loss: %.8f, Evaluation AUC:  %.8f\n" % (averaged_eval_loss, eval_auc) 
            logfile.write(line)
        logfile.close()


    def print_evaluation_after_retraining(self):
        test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.final_evaluation, self.anno, dropout=True)
        eval_total_loss=[]
        eval_total_preds=[]
        for j_ in range(self.num_sampling):
            loss, preds = self.sess.run(
                                        [self.total_loss, 
                                         self.preds], 
                                         feed_dict=test_feed_dict,
                                        )
            eval_total_loss.append(loss)
            eval_total_preds.append(preds)

        averaged_eval_loss = np.mean(eval_total_loss, axis=0)
        total_preds = np.mean(eval_total_preds, axis=0)

        eval_roc, eval_auc = ROC_AUC(total_preds, test_feed_dict[self.labels_placeholder])
        cprint("[*] Evaluation loss: %.8f, Evaluation AUC:  %.8f" % (averaged_eval_loss, eval_auc), bg_color='r')
        return (averaged_eval_loss, eval_auc)


