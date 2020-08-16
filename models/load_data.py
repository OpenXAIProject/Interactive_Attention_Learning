import numpy as np
from models.dataset import DataSet
import pickle
import pdb

def Load_Data(task, data_path):
    print('Task: %s' %(task))
    with open(data_path+'/'+'processed_datasets.pkl', 'rb') as fp:
        dataset = pickle.load(fp)

    feature_names = np.array(dataset['feature_names'])
    train_inputs = np.array(dataset['train_inputs'])
    train_labels = np.array(dataset['train_labels'])
    validation_inputs=np.array(dataset['validation_inputs'])
    validation_labels= np.array(dataset['validation_labels'])
    evaluation_inputs = np.array(dataset['evaluation_inputs'])
    evaluation_labels = np.array(dataset['evaluation_labels'])

    train = DataSet(train_inputs, train_labels)
    validation = DataSet(validation_inputs, validation_labels)
    evaluation = DataSet(evaluation_inputs, evaluation_labels)

    return train, validation, evaluation

def Baseline_version_load_Data(task, data_path):
    print('Task: %s' %(task))
    with open(data_path+'/'+'processed_datasets.pkl', 'rb') as fp:
        dataset = pickle.load(fp)

    feature_names = np.array(dataset['feature_names'])
    train_inputs = np.array(dataset['train_inputs'])
    train_labels = np.array(dataset['train_labels'])
    validation_inputs=np.array(dataset['validation_inputs'])
    validation_labels= np.array(dataset['validation_labels'])
    evaluation_inputs = np.array(dataset['evaluation_inputs'])
    evaluation_labels = np.array(dataset['evaluation_labels'])
    
    train_inputs=train_inputs
    train_labels=train_labels
    validation_inputs=validation_inputs
    validation_labels=validation_labels
    evaluation_inputs = evaluation_inputs
    evaluation_labels = evaluation_labels
  
    train_inputs = np.append(train_inputs, validation_inputs, axis=0)
    train_labels = np.append(train_labels, validation_labels, axis=0)
    evaluation_inputs = evaluation_inputs
    evaluation_labels = evaluation_labels

    train = DataSet(train_inputs, train_labels)
    evaluation = DataSet(evaluation_inputs, evaluation_labels)

    return train, evaluation

