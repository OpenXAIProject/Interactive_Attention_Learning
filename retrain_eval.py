from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import tensorflow as tf

import  models.experiments as experiments
from models.model import retain_np
from models.load_data import Load_Data, Baseline_version_load_Data
import collections
import pdb
from models.anno_load_data import *
import pickle
from models.dataset import DataSet

task = 'cerebral1'

data_path = '/Data/EHR_Datasets/'+ task
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'final_evaluation'])
Datasets.train, Datasets.test, Datasets.final_evaluation = Load_Data(task, data_path)    

#ANNO set
Anno = collections.namedtuple('Anno', ['zero_alpha', 'zero_beta', 'alpha', 'beta', 'annotated_input', 'annotated_labels'])
Anno.zero_alpha, Anno.zero_beta, Anno.alpha, Anno.beta, Anno.annotated_input, Anno.annotated_labels = anno_load_data(
    task, '/IAL_Cerabral_Infartion/annotation/', '1')


#Annoated data points
Anno_Datasets = collections.namedtuple('Anno_Datasets', ['retrain'])
Anno_Datasets.retrain = DataSet(annotated_sample_inputs, annotated_sample_labels)

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

model = retain_np(
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
                  num_sampling=50,
                  train_dir='output',
                  log_dir = 'log',
                  model_name ='source_code'
                  )


test_idx = np.arange(Datasets.test.labels.shape[0])

initial_iter_to_load = num_steps-1
retrain_num_steps=70 
retrain_iter_to_load = 41


model.retrain_train_instance_after_oracle(
                                          iter_to_load=retrain_iter_to_load,                                          
                                          num_steps=retrain_num_steps,
                                          anno=Anno,
                                          retrain_dir='retrain_output',
                                          anno_dataset=Anno_Datasets,
                                          save_checkpoints=True,
                                          verbose=True,
                                          dropout=True,
                                          do_checks=True,
                                         )
