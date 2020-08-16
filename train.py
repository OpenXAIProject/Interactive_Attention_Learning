
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import numpy as np
import IPython
import tensorflow as tf

import models.experiments as experiments
import models.experiments_ifif as experiments_ifif
import models.experiments_counterfactual as experiments_counterfactual
import models.experiments_ut_counterfactual as experiments_ut_counterfactual

from models.model import IAL_NAP
from models.load_data import Load_Data, Baseline_version_load_Data
import collections
import pdb
from models.anno_load_data import *

task = 'cerebral1'
stage= 's1_'
learning_stage = 's2_'
data_path = './data/EHR/' + task


Datasets = collections.namedtuple('Datasets', ['train', 'test', 'final_evaluation'])
Datasets.train, Datasets.test, Datasets.final_evaluation = Load_Data(task, data_path)    

print("Data Size")
print('train: %s' %(Datasets.train.x.shape[0]))
print('validation: %s' %(Datasets.test.x.shape[0]))
print('evaluation: %s' %(Datasets.final_evaluation.x.shape[0]))

#ANNO set
Anno = collections.namedtuple('Anno', ['alpha', 'beta', 'annotated_input', 'annotated_label'])
Anno.alpha, Anno.beta, Anno.annotated_input, Anno.annotated_label = anno_load_data(task, './annotation/', stage='initial')

#Data info
num_features = Datasets.train.x.shape[2]
steps = Datasets.train.x.shape[1]
num_test_points = Datasets.test.num_examples 
num_train_points = Datasets.train.num_examples

# Model info
num_layers = 1
hidden_units = 34
embed_size = 34
alpha_np_shape=hidden_units+1
np_shape=hidden_units*2
np_batch=40

batch_size = 200
num_sampling = 30
weight_decay = 0.001

initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]

input_keep_probs = 0.8
output_keep_probs = 0.8
state_keep_probs = 0.8
num_steps = 10000

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

model.train(
            num_steps=num_steps, 
            dropout=True,
           )

iter_to_load = num_steps - 1
samples =experiments_ut_counterfactual.Find_influential_training_input(
                                                          model, 
                                                          iter_to_load=iter_to_load,
                                                          force_refresh=False, 
                                                          num_to_choose_test=400,
                                                          num_to_choose_train=10,                                                 
                                                          num_steps=1000,
                                                          random_seed=17,
                                                          remove_type='maxinf',
                                                          model_name=stage+'source_code',
                                                          approx_type='cg',
                                                          loss_type='normal_loss',
                                                          test_description="training_indices",
                                                          train_dir=stage+'output',
                                                          num_sampling=30,
                                                          task=task,
                                                          stage=stage, 
                                                        )

# Call Web Annotation System
# os.system('python ./hil_medical_annotator/main.py')
