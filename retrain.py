
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import tensorflow as tf

import models.experiments as experiments
from models.model import IAL_NAP
from models.load_data import Load_Data, Baseline_version_load_Data
import collections
import pdb
from models.anno_load_data import *
import pickle
from models.dataset import DataSet

task = 'cerebral1'
stage= 's1_'
learning_stage = 's2_'
data_path = './data/EHR/' + task

Datasets = collections.namedtuple('Datasets', ['train', 'test', 'final_evaluation'])
Datasets.train, Datasets.test, Datasets.final_evaluation = Load_Data(task, data_path)    

#ANNO set
Anno = collections.namedtuple('Anno', ['alpha', 'beta', 'annotated_input', 'annotated_label'])
Anno.alpha, Anno.beta, annotated_input, annotated_label = anno_load_data(task, './annotation/', stage=stage)

#Annoated data points
Anno_Datasets = collections.namedtuple('Anno_Datasets', ['retrain'])
Anno_Datasets.retrain = DataSet(annotated_input, annotated_label)

#Data info
num_features = Anno_Datasets.retrain.x.shape[2]
steps = Anno_Datasets.retrain.x.shape[1]

num_test_points = Datasets.test.num_examples 
num_train_points = Datasets.train.num_examples

# Model info
num_layers = 1
hidden_units = 34
embed_size = 34
alpha_np_shape=hidden_units+1
np_shape=hidden_units*2

np_batch=20
batch_size = 20

num_sampling = 30
weight_decay = 0.001

initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]

input_keep_probs = 0.8
output_keep_probs = 0.8
state_keep_probs = 0.8
num_steps = 10000


test_idx = np.arange(Datasets.test.labels.shape[0])
iter_to_load = num_steps-1
retrain_num_steps=50

model = IAL_NAP(
                num_features=num_features,
                steps=steps,
                num_layers=num_layers,
                hidden_units=hidden_units,
                embed_size=embed_size,
                alpha_np_shape=alpha_np_shape,
                np_shape=np_shape,
                np_batch=np_batch,
                n_batch_size=batch_size,
                num_test_points=num_test_points,
                num_train_points=num_train_points,
                batch_size=batch_size,
                data_sets=Datasets,
                anno=Anno,
                initial_learning_rate=initial_learning_rate,
                weight_decay=weight_decay,
                damping=1e-2,
                decay_epochs=decay_epochs,
                mini_batch=True,
                input_keep_probs=input_keep_probs,
                output_keep_probs=output_keep_probs,
                state_keep_probs=state_keep_probs,
                variational_dropout=False,
                num_sampling=30,
                train_dir=stage+'output',
                retrain_dir=learning_stage+'output',
                log_dir = 'log',
                model_name =stage+'source_code',
                retrain_model_name=learning_stage+'source_code',
               )


model.retrain_train_instance_after_oracle(
                                          iter_to_load=iter_to_load,
                                          num_steps=retrain_num_steps,
                                          anno=Anno,
                                          retrain_dir=learning_stage+'output', 
                                          anno_dataset=Anno_Datasets,
                                          save_checkpoints=True,
                                          verbose=True,
                                          dropout=True,
                                          do_checks=True,
                                         )


sampling_iter_to_load = retrain_num_steps-1
samples =experiments.further_stage_Find_influential_training_input(
                                                                  model, 
                                                                  iter_to_load=sampling_iter_to_load,
                                                                  force_refresh=False, 
                                                                  num_to_choose_test=400,
                                                                  num_to_choose_train=100,
                                                                  num_steps=1000,
                                                                  random_seed=17,
                                                                  remove_type='maxinf',
                                                                  model_name=learning_stage+'source_code',
                                                                  approx_type='cg',
                                                                  loss_type='normal_loss',
                                                                  test_description=learning_stage+"training_indices",
                                                                  train_dir=learning_stage+'output',
                                                                  num_sampling=30,
                                                                  task=task,
                                                                  stage=learning_stage,
                                                                  )
