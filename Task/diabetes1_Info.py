import numpy as np
import pdb

def task_info(task):
    task_title = task
    data_path = '/Data/EHR_Datasets/' + task_title
    idx_collected = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    new_idx = [0,1,2,3,4,5,6,12,13,14,15,23,24,25,26,27,29,30,31,32]
    
    idx_features = new_idx
    print(task)
    print('Collected features for:')
    print(idx_features)
    return task_title, data_path, idx_features
