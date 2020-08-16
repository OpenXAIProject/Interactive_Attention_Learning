
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import tensorflow as tf

import  models.experiments as experiments
from models.All_UA import All_UA
from models.load_data import Load_Data
import collections
import pdb
import pickle

task = 'cerebral1'
data_path = '/Data/EHR_Datasets/'+ task
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'final_evaluation'])

Datasets.train, Datasets.test, Datasets.final_evaluation = Load_Data(task, data_path)    

#Data info
num_features = Datasets.train.x.shape[2]
steps = Datasets.train.x.shape[1]

# Model info
num_layers = 1
hidden_units = 34
embed_size = 34
batch_size = 200
num_sampling = 30

weight_decay = 0.001
batch_size = 500
initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]

input_keep_probs = 0.7
output_keep_probs = 0.7
state_keep_probs = 0.7


#Call masks 
numpy_filename = task + "_Samples.npy"
masks = np.load(numpy_filename, 'r')

model = All_UA(
                masks=None,
                num_features=num_features,
                steps=steps,
                num_layers=num_layers,
                hidden_units=hidden_units,
                embed_size=embed_size,
                batch_size=batch_size,
                data_sets=Datasets,
                initial_learning_rate=initial_learning_rate,
                weight_decay=weight_decay,
                damping=1e-2,
                decay_epochs=decay_epochs,
                mini_batch=True,
                input_keep_probs=input_keep_probs,
                output_keep_probs=output_keep_probs,
                state_keep_probs=state_keep_probs,
                variational_dropout=False,
                num_sampling=100,
                train_dir='output',
                retrain_dir='Retrain_Output',
                log_dir = 'log',
                model_name ='source_code'
               )


num_steps=5000

test_idx = np.arange(Datasets.test.labels.shape[0])

#Retrain with adjusted training sets 
original_test_indices_file_name = task + "_Samples.pickle"
with open(original_test_indices_file_name, 'rb') as handle:
    sets = pickle.load(handle)

input_feed = sets["RealValuesInput"]
labels_feed = sets["RealValuesLabels"]


iter_to_load = 49999
retrain_num_steps= 30
iter_to_load=retrain_num_steps-1

for i_ in range(30):
  print('i_th')
  model.print_evaluation_after_retraining(i_)


