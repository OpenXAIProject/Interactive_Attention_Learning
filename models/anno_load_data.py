import numpy as np
from models.anno import Anno
import pickle
import os
import pdb


def anno_load_data(task, data_path, stage):
    print('Task for Annotation training: %s' % (task))
    if stage == 'initial':
        sample = np.load(os.path.join(data_path, '%s_Samples.npy' % task))
        alpha_annotations = np.zeros([sample.shape[0], sample.shape[1], 1])
        beta_annotations = np.zeros([sample.shape[0], sample.shape[1], sample.shape[2]]) 

        neg_one_alpha = np.negative(np.ones([alpha_annotations.shape[0], alpha_annotations.shape[1], 1]))
        neg_one_beta = np.negative(np.ones([beta_annotations.shape[0], beta_annotations.shape[1], beta_annotations.shape[2]]))

        alpha = Anno(neg_one_alpha)
        beta = Anno(neg_one_beta)
        
        annotated_input=Anno(neg_one_alpha) 
        annotated_label=Anno(neg_one_beta)

    else: 
        annotations = np.load(os.path.join(data_path, '%s%s_Samples.npy' % (stage, task)))
        beta = annotations
        alpha = np.ones([annotations.shape[0], annotations.shape[1], 1])

        beta = Anno(beta)
        alpha = Anno(alpha)

        file = open('./annotation/Samples/' + '%s%s_Samples.pickle' % (stage, task), 'rb')
        sample = pickle.load(file)
        annotated_input = sample['RealValuesInput']
        annotated_label = sample['RealValuesLabels']
        
    return alpha, beta, annotated_input, annotated_label
