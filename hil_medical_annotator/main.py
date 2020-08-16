import re
import os
import sys
import numpy as np
import pickle
import random
import math
import socket
import time
import math
from datetime import datetime
from flask import Flask, request, session, url_for, redirect, render_template

from utils.preprocess import *
from utils.feature_name import feature_name_list, feature_unit_list
from utils.print_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Initialize Flask
app = Flask(__name__)

###############################################################################
# Set disease type
###############################################################################
# disease_list = ['s1_cardiovascular1']
disease_list = ['s1_cerebral1']

Infl_pickle_list = []
for disease in disease_list:
    Infl_pickle_list.append('%s_Samples' % disease.lower())


###############################################################################
# MAIN CONFIG
###############################################################################
DATASET_TYPE   = 'EHR'  # NEED TO REVISE
MAIN_ROOT      = './data/%s' % DATASET_TYPE
TIME_STEP      = 4
FEATURE_NUMBER = 34

###############################################################################
# DETAIL CONFIG
###############################################################################
CONFIG_NAME       = 'CF'             # NEED TO REVISE
VIS_FEATURE_COUNT = 8                # NEED TO REVISE
CONFIG_NAME = CONFIG_NAME + '_FEAT%s' % VIS_FEATURE_COUNT

# Graph count to visualize
GRAPH_COUNT = int(math.ceil(VIS_FEATURE_COUNT / 11))
graph_mask = [False, False, False]
for i in range(GRAPH_COUNT):
    graph_mask[i] = True


# Path Info
# root_path   = './data/%s/%s' % (DATASET_TYPE, CONFIG_NAME)
# result_path = os.path.join(root_path, 'results')
root_path   = './annotation'
result_path = root_path
sample_path = os.path.join(root_path, 'Samples')
server_storage_path = result_path

# Check Sample Path
if not os.path.exists(sample_path):
    print_error_message("=> No sample path found at {}".format(sample_path))

if not os.path.exists(result_path):
    os.makedirs(result_path)


###############################################################################
# Reading Data
###############################################################################
Infl_data_list = []
for i in range(len(Infl_pickle_list)):
    Infl_pickle_path = os.path.join(sample_path, '%s.pickle' % Infl_pickle_list[i])
    with open(Infl_pickle_path, 'rb') as f0:
        Infl_data = pickle.load(f0)
    Infl_data_list.append(Infl_data)
 

###############################################################################
# Writing Result
###############################################################################
num_per_sample = 10
num_of_disease = len(disease_list)
total_n = num_per_sample * num_of_disease


# Default Partition
sample_partition = 10


# check STAGE INDEX
initial_disease_name = disease_list[0]
initial_stage_index  = initial_disease_name.split('_')[0]
assert(initial_stage_index in ['s1', 's2', 's3', 's4', 's5', 's6'])


# Set fixed user_name
# user_name = '%s_disease_%d_%d' % (initial_stage_index, sample_partition - num_per_sample, sample_partition - 1)
user_name = './'
user_path = os.path.join(server_storage_path, user_name)
if not os.path.exists(user_path):
    os.makedirs(user_path)
# if not os.path.exists(os.path.join(user_path, 'Explanations')):
#     os.makedirs(os.path.join(user_path, 'Explanations'))
# if not os.path.exists(os.path.join(user_path, 'Times')):
#     os.makedirs(os.path.join(user_path, 'Times'))

global_user_name = user_name


###############################################################################
# Output
###############################################################################
Infl_output = np.ones((num_of_disease, num_per_sample, TIME_STEP, FEATURE_NUMBER), dtype=np.float32)

# load data if exists
for i in range(len(disease_list)):
    result_name = os.path.join(server_storage_path, global_user_name, '%s_Samples.npy' % (disease_list[i].lower()))
    if os.path.exists(result_name):
        Infl_output[i] = np.load(result_name)


###############################################################################
# Times Output
###############################################################################
Times_output = np.zeros((num_of_disease, num_per_sample, TIME_STEP), dtype=np.float32)

# load time if exists:
for i in range(len(disease_list)):
    # time_result_name = os.path.join(server_storage_path, global_user_name, 'Times', '%s_Samples_Times.npy' % (disease_list[i].lower()))
    time_result_name = os.path.join(server_storage_path, global_user_name, '%s_Samples_Times.npy' % (disease_list[i].lower()))
    if os.path.exists(time_result_name):
        Times_output[i] = np.load(time_result_name)


# Use more global values
patient_id    = None
patient_info  = None
patient_year = None 

# for patient year
year_button_idx = 0
hide_features   = None
cf_features     = np.ones((VIS_FEATURE_COUNT), dtype=np.float32)


# check spending time for annotation
stopwatch_start = None

# For Counterfactual Estimation
cf_feature_name  = ""
cf_feature_value = ""
cf_value1 = 0.
cf_value2 = 0.

###############################################################################
# NEED TO MERGE WITH MODEL TO GET ACTUAL ESTIMATION ŷ
###############################################################################
cf_values = np.random.rand(total_n)


def non_hide_func(read_all_feature=False):
    global patient_year
    global patient_id
    global patient_info

    global year_button_idx
    global hide_features
    global stopwatch_start
    global graph_mask

    global cf_features
    global cf_feature_name
    global cf_feature_value
    global cf_value1
    global cf_value2


    if request.method == 'POST':
        # recall python url
        n = int(request.form['n'])

        # Bring checked form of checkbox
        hide_idx_list = list(map(lambda x: int(x), request.form.getlist('hide')))

        # Omit [, ]
        hide_features = str(hide_idx_list)[1:-1]
    else:
        n = int(request.args.get('n'))
        hide_idx_list = None


    # call reverse index (sorted by data['InfluenceScores'] or data['PredictionUncertainty'])
    n = min(max(0, n), total_n - 1)

    # go to first element of each disease
    # n = n - (n % num_per_sample)
    row = n // num_per_sample
    col = n % num_per_sample

    # Only Infl setting
    data = Infl_data_list[row]
    output = Infl_output[row]

    # small -> large Influence
    n_th_patient = sample_partition - col - 1
    disease_name = disease_list[row]

    # main information from pickle & reset button idx
    if not read_all_feature:
        year_button_idx = 0
    patient_id      = n_th_patient
    patient_label   = int(data['RealValuesLabels'][n_th_patient][0])

    total_year_exists = 0
    for i in range(len(data['FeatureContribution'][n_th_patient])):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            total_year_exists += 1

    patient_info = preprocess_real_inputs(data['RealValuesInput'][n_th_patient][total_year_exists - year_button_idx - 1])

    # call FeatureContribution
    feature_contrib_list = data['FeatureContribution'][n_th_patient][total_year_exists - year_button_idx - 1]
    abs_feature_contrib_list = np.absolute(feature_contrib_list)

    # get rank by unglobal cf_value1
    global cf_value2certainty
    fc_uncertainty = data['FC_Uncertainty'][n_th_patient][total_year_exists - year_button_idx - 1]
    uncertainty_ranks = fc_uncertainty.argsort()[::-1]

    # sort feature_name_list & feature_contrib_list
    np_patient_info        = np.array(patient_info.copy())
    np_feature_unit_list    = np.array(feature_unit_list.copy())
    np_feature_name_list    = np.array(feature_name_list.copy())
    np_feature_contrib_list = np.array(abs_feature_contrib_list.copy())

    # Visualize Top VIS_FEATURE_COUNT Uncertain Feature
    feature_contrib_max = 1.0
    if read_all_feature:
        uncertainty_ranked_patient_info = np_patient_info[uncertainty_ranks].tolist()
        uncertainty_ranked_feature_unit_list = np_feature_unit_list[uncertainty_ranks].tolist()
        uncertainty_ranked_feature_name_list = np_feature_name_list[uncertainty_ranks].tolist()
        uncertainty_ranked_feature_contrib_list = np_feature_contrib_list[uncertainty_ranks].tolist()
        uncertainty_ranked_feature_contrib_sum = sum(uncertainty_ranked_feature_contrib_list)
        uncertainty_ranked_feature_contrib_percentage = [x/uncertainty_ranked_feature_contrib_sum for x in uncertainty_ranked_feature_contrib_list]
    else:
        uncertainty_ranked_patient_info = np_patient_info[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
        uncertainty_ranked_feature_unit_list = np_feature_unit_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
        uncertainty_ranked_feature_name_list = np_feature_name_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
        uncertainty_ranked_feature_contrib_list = np_feature_contrib_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
        uncertainty_ranked_feature_contrib_sum = sum(uncertainty_ranked_feature_contrib_list)
        uncertainty_ranked_feature_contrib_percentage = [x/uncertainty_ranked_feature_contrib_sum for x in uncertainty_ranked_feature_contrib_list]


    # call Feature is Checked
    checkbox_list = [''] * FEATURE_NUMBER
    current_output = output[n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][uncertainty_ranks]
    for i in range(len(checkbox_list)):
        if current_output[i] == 0.:
            checkbox_list[i] = 'checked'


    # call CF is Checked
    cf_features[:] = 1.
    cf_checkbox_list = [''] * VIS_FEATURE_COUNT
    for i in range(len(cf_features)):
        if cf_features[i] == 0.:
            cf_checkbox_list[i] = 'checked'


    # collect train_year of PID
    current_year_str = ['', '', '', '']
    current_year_act = ['disabled', 'disabled', 'disabled', 'disabled']
    old_age = int(data['RealValuesInput'][n_th_patient][total_year_exists-1][1])
    offset = TIME_STEP - total_year_exists
    for i in range(data['FeatureContribution'][n_th_patient].shape[0]):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            current_age = int(data['RealValuesInput'][n_th_patient][i][1])
            current_year_str[offset + i] = 2012 - old_age + current_age
            current_year_act[offset + i] = None


    # load comments
    patient_year = current_year_str[-year_button_idx-1]
    txtbox_1, txtbox_2, txtbox_3 = load_textboxes(server_storage_path, global_user_name, disease_name, patient_id, patient_year)


    cf_feature_name  = ""
    cf_feature_value = ""
    cf_value1 = cf_values[n]
    cf_value2 = 0.


    return (
             n,
             patient_year,
             patient_id,
             uncertainty_ranked_patient_info,
             uncertainty_ranked_feature_unit_list,
             patient_label,
             disease_name,
             num_per_sample,
             checkbox_list,
             txtbox_1,
             txtbox_2,
             txtbox_3,
             uncertainty_ranked_feature_name_list,
             uncertainty_ranked_feature_contrib_percentage,
             feature_contrib_max,
             current_year_str,
             current_year_act,
             graph_mask,
             cf_checkbox_list,
             cf_feature_name,
             cf_feature_value,
             cf_value1,
             cf_value2,
           )



# 'app.route' tells whar URL should trigger our function
@app.route('/read_non_hide', methods=['GET', 'POST'])
def read_non_hide():
    (n,
     patient_year,
     patient_id,
     uncertainty_ranked_patient_info,
     uncertainty_ranked_feature_unit_list,
     patient_label,
     disease_name,
     num_per_sample,
     checkbox_list,
     txtbox_1,
     txtbox_2,
     txtbox_3,
     uncertainty_ranked_feature_name_list,
     uncertainty_ranked_feature_contrib_list,
     feature_contrib_max,
     current_year_str,
     current_year_act,
     graph_mask,
     cf_checkbox_list,
     cf_feature_name,
     cf_feature_value,
     cf_value1,
     cf_value2,
    ) = non_hide_func(read_all_feature=True)

    return render_template('no_hide.html',
        n = n,
        patient_year  = patient_year,
        patient_id   = patient_id,
        patient_info = uncertainty_ranked_patient_info,
        feature_unit = uncertainty_ranked_feature_unit_list,
        patient_label = patient_label,
        disease_name = disease_name,
        disease_offset = num_per_sample,
        checkbox_list = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list=uncertainty_ranked_feature_name_list,
        feature_contrib_list=uncertainty_ranked_feature_contrib_list,
        feature_contrib_max=feature_contrib_max,
        current_year_str = current_year_str,
        current_year_act = current_year_act,

        graph_mask                 = [True] * len(graph_mask),
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )




# 'app.route' tells whar URL should trigger our function
@app.route('/disease_search', methods=['GET', 'POST'])
def disease_search():
    (n,
     patient_year,
     patient_id,
     uncertainty_ranked_patient_info,
     uncertainty_ranked_feature_unit_list,
     patient_label,
     disease_name,
     num_per_sample,
     checkbox_list,
     txtbox_1,
     txtbox_2,
     txtbox_3,
     uncertainty_ranked_feature_name_list,
     uncertainty_ranked_feature_contrib_list,
     feature_contrib_max,
     current_year_str,
     current_year_act,
     graph_mask,

     cf_checkbox_list,
     cf_feature_name,
     cf_feature_value,
     cf_value1,
     cf_value2,
    ) = non_hide_func()
    

    return render_template('no_hide.html',
        n = n,
        patient_year  = patient_year,
        patient_id   = patient_id,
        patient_info = uncertainty_ranked_patient_info,
        feature_unit = uncertainty_ranked_feature_unit_list,
        patient_label = patient_label,
        disease_name = disease_name,
        disease_offset = num_per_sample,
        checkbox_list = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list=uncertainty_ranked_feature_name_list,
        feature_contrib_list=uncertainty_ranked_feature_contrib_list,
        feature_contrib_max=feature_contrib_max,
        current_year_str = current_year_str,
        current_year_act = current_year_act,

        graph_mask = graph_mask,
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )



# 'app.route' tells what URL should trigger our function
@app.route('/patient_search', methods=['GET', 'POST'])
def patient_search():
    (n,
     patient_year,
     patient_id,
     uncertainty_ranked_patient_info,
     uncertainty_ranked_feature_unit_list,
     patient_label,
     disease_name,
     num_per_sample,
     checkbox_list,
     txtbox_1,
     txtbox_2,
     txtbox_3,
     uncertainty_ranked_feature_name_list,
     uncertainty_ranked_feature_contrib_list,
     feature_contrib_max,
     current_year_str,
     current_year_act,
     graph_mask,
     cf_checkbox_list,
     cf_feature_name,
     cf_feature_value,
     cf_value1,
     cf_value2,
    ) = non_hide_func()


    return render_template('no_hide.html',
        n = n,            
        patient_year   = patient_year,
        patient_id     = patient_id,
        patient_info   = uncertainty_ranked_patient_info,
        feature_unit   = uncertainty_ranked_feature_unit_list,
        patient_label  = patient_label,
        disease_name   = disease_name,
        disease_offset = num_per_sample, 
        checkbox_list  = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list    = uncertainty_ranked_feature_name_list,
        feature_contrib_list = uncertainty_ranked_feature_contrib_list,
        feature_contrib_max  = feature_contrib_max,
        current_year_str = current_year_str,
        current_year_act = current_year_act,

        graph_mask = graph_mask,
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )
       


# 'appp.route' tells what URL should trigger our function
@app.route('/start_check', methods=['GET', 'POST'])
def start_check():
    global patient_year
    global patient_id
    global patient_info

    global year_button_idx
    global hide_features
    global stopwatch_start
    global graph_mask

    global cf_features
    global cf_feature_name
    global cf_feature_value
    global cf_value1
    global cf_value2


    if request.method == 'POST':
        # recall python url
        n = int(request.form['n'])

        # Bring checked form of checkbox
        hide_idx_list = list(map(lambda x: int(x), request.form.getlist('hide')))

        # Omit [, ]
        hide_features = str(hide_idx_list)[1:-1]
    else:
        n = int(request.args.get('n'))
        hide_idx_list = None

   
    # start stopwatch 
    stopwatch_start = time.time()

    # call reverse index (sorted by data['InfluenceScores'] or data['PredictionUncertainty'])
    n = min(max(0, n), total_n - 1)
    row = n // num_per_sample
    col = n % num_per_sample

    # Only Infl setting
    data = Infl_data_list[row]
    output = Infl_output[row]

    # small -> large Influence
    n_th_patient = sample_partition - col - 1
    disease_name = disease_list[row] 
    patient_id    = n_th_patient
    patient_label = int(data['RealValuesLabels'][n_th_patient][0])


    total_year_exists = 0
    for i in range(len(data['FeatureContribution'][n_th_patient])):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            total_year_exists += 1

    patient_info = preprocess_real_inputs(data['RealValuesInput'][n_th_patient][total_year_exists - year_button_idx - 1])


    # call FeatureContribution
    feature_contrib_list = data['FeatureContribution'][n_th_patient][total_year_exists - year_button_idx - 1]
    abs_feature_contrib_list = np.absolute(feature_contrib_list)

    # get rank by uncertainty
    fc_uncertainty = data['FC_Uncertainty'][n_th_patient][total_year_exists - year_button_idx - 1]
    uncertainty_ranks = fc_uncertainty.argsort()[::-1]

    # sort feature_name_list & feature_contrib_list
    np_patient_info        = np.array(patient_info.copy())
    np_feature_unit_list    = np.array(feature_unit_list.copy())
    np_feature_name_list    = np.array(feature_name_list.copy())
    np_feature_contrib_list = np.array(abs_feature_contrib_list.copy())

    
    # Visualize Top VIS_FEATURE_COUNT Uncertain Feature
    uncertainty_ranked_patient_info = np_patient_info[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_unit_list = np_feature_unit_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_name_list = np_feature_name_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_list = np_feature_contrib_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_sum = sum(uncertainty_ranked_feature_contrib_list)
    uncertainty_ranked_feature_contrib_percentage = [x/uncertainty_ranked_feature_contrib_sum for x in uncertainty_ranked_feature_contrib_list]
    feature_contrib_max = 1.0

    # call Feature is Checked
    checkbox_list = [''] * FEATURE_NUMBER
    current_output = output[n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][uncertainty_ranks]
    for i in range(len(checkbox_list)):
        if current_output[i] == 0.:
            checkbox_list[i] = 'checked'


    # call CF is Checked
    cf_checkbox_list = [''] * VIS_FEATURE_COUNT
    for i in range(len(cf_features)):
        if cf_features[i] == 0.:
            cf_checkbox_list[i] = 'checked'


    # collect train_year of PID
    current_year_str = ['', '', '', '']
    current_year_act = ['disabled', 'disabled', 'disabled', 'disabled']
    old_age = int(data['RealValuesInput'][n_th_patient][total_year_exists-1][1])
    offset = TIME_STEP - total_year_exists
    for i in range(data['FeatureContribution'][n_th_patient].shape[0]):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            current_age = int(data['RealValuesInput'][n_th_patient][i][1])
            current_year_str[offset + i] = 2012 - old_age + current_age
            current_year_act[offset + i] = None


    # load comments
    patient_year = current_year_str[-year_button_idx-1]
    txtbox_1, txtbox_2, txtbox_3 = load_textboxes(server_storage_path, global_user_name, disease_name, patient_id, patient_year)

    cf_value1 = cf_values[n]
    cf_value2 = 0.


    return render_template('hide.html',
        n = n,
        patient_year  = patient_year,
        patient_id   = patient_id,
        patient_info = uncertainty_ranked_patient_info,
        feature_unit = uncertainty_ranked_feature_unit_list,
        patient_label = patient_label,
        disease_name = disease_name,
        disease_offset = num_per_sample,
        checkbox_list = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list=uncertainty_ranked_feature_name_list,
        feature_contrib_list=uncertainty_ranked_feature_contrib_percentage,
        feature_contrib_max=feature_contrib_max,

        current_year_str = current_year_str,
        current_year_act = current_year_act,

        graph_mask = graph_mask,
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )



 

# 'appp.route' tells what URL should trigger our function
@app.route('/save_result', methods=['GET', 'POST'])
def save_result():
    global patient_year
    global patient_id
    global patient_info

    global year_button_idx
    global hide_features
    global stopwatch_start
    global output
    global graph_mask

    global cf_features
    global cf_feature_name
    global cf_feature_value
    global cf_value1
    global cf_value2


    if request.method == 'POST':
        # recall python url
        n = int(request.form['n'])

        # Bring checked form of checkbox
        hide_idx_list = list(map(lambda x: int(x), request.form.getlist('hide')))

        # Omit [, ]
        hide_features = str(hide_idx_list)[1:-1]

        # Read comment 
        text1 = request.form['text1']
    else:
        n = int(request.args.get('n'))
        hide_idx_list = None
        text1 = None

    # stop stopwatch once
    stopwatch_duration = time.time() - stopwatch_start 
    print('stopwatch_duration :', stopwatch_duration)
    print('')

    # call reverse index (sorted by data['InfluenceScores'] or data['PredictionUncertainty'])
    n = min(max(0, n), total_n - 1)
    row = n // num_per_sample
    col = n % num_per_sample

    # Only Infl setting
    data   = Infl_data_list[row]
    output = Infl_output[row]

    # small -> large Influence
    n_th_patient  = sample_partition - col - 1
    disease_name  = disease_list[row]
    patient_id    = n_th_patient
    patient_label = int(data['RealValuesLabels'][n_th_patient][0])


    total_year_exists = 0
    for i in range(len(data['FeatureContribution'][n_th_patient])):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            total_year_exists += 1

    patient_info = preprocess_real_inputs(data['RealValuesInput'][n_th_patient][total_year_exists - year_button_idx - 1]) 


    # call FeatureContribution
    feature_contrib_list = data['FeatureContribution'][n_th_patient][total_year_exists - year_button_idx - 1]
    abs_feature_contrib_list = np.absolute(feature_contrib_list)

    # get rank by uncertainty
    fc_uncertainty = data['FC_Uncertainty'][n_th_patient][total_year_exists - year_button_idx - 1]
    uncertainty_ranks = fc_uncertainty.argsort()[::-1]

    # sort feature_name_list & feature_contrib_list
    np_patient_info        = np.array(patient_info.copy())
    np_feature_unit_list    = np.array(feature_unit_list.copy())
    np_feature_name_list    = np.array(feature_name_list.copy())
    np_feature_contrib_list = np.array(abs_feature_contrib_list.copy())


    # Visualize Top-VIS_FEATURE_COUNT Uncertain Feature
    uncertainty_ranked_patient_info = np_patient_info[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_unit_list = np_feature_unit_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_name_list = np_feature_name_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_list = np_feature_contrib_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_sum = sum(uncertainty_ranked_feature_contrib_list)
    uncertainty_ranked_feature_contrib_percentage = [x/uncertainty_ranked_feature_contrib_sum for x in uncertainty_ranked_feature_contrib_list]
    feature_contrib_max = 1.0


    # collect train_year of PID
    current_year_str = ['', '', '', '']
    current_year_act = ['disabled', 'disabled', 'disabled', 'disabled']
    old_age = int(data['RealValuesInput'][n_th_patient][total_year_exists-1][1])
    offset = TIME_STEP - total_year_exists
    for i in range(data['FeatureContribution'][n_th_patient].shape[0]):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            current_age = int(data['RealValuesInput'][n_th_patient][i][1])
            current_year_str[offset + i] = 2012 - old_age + current_age
            current_year_act[offset + i] = None



    patient_year = current_year_str[-year_button_idx-1]

    # PDB_TRACING -> hide_idx_list
    print('\nrequest.method', request.method)
    print('hide_idx_list', hide_idx_list)
    print('')


    # save binary mask [num_per_sample, TIME_STEP, FEATURE_NUMBER] as numpy
    result_name = os.path.join(server_storage_path, global_user_name, '%s_Samples.npy' % (disease_name.lower()))
    Infl_output[row][n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][:] = 1.
    for hf in hide_idx_list:
        Infl_output[row][n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][uncertainty_ranks[int(hf) - 1]] = 0.
    np.save(result_name, Infl_output[row])

    # save comment
    txt_result_name_1 = os.path.join(server_storage_path, global_user_name, 'Explanations', '%s_%.12d_%d_1.txt' % ('%s_Samples' % disease_name.lower(), patient_id, patient_year))

    if len(str(text1)) > 0:
        line = str(text1) + '\n'
        with open(txt_result_name_1, 'wt') as f:
            f.write(line)

    # save time
    time_result_name = os.path.join(server_storage_path, global_user_name, '%s_Samples_Times.npy' % (disease_name.lower()))
    Times_output[row][n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1] += stopwatch_duration
    np.save(time_result_name, Times_output[row])


    # Sync output = Infl_output    
    output = Infl_output[row]

    # call Feature is Checked
    checkbox_list = [''] * FEATURE_NUMBER 
    current_output = output[n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][uncertainty_ranks]
    for i in range(len(checkbox_list)):
        if current_output[i] == 0.:
            checkbox_list[i] = 'checked'

    # call CF is Checked
    cf_checkbox_list = [''] * VIS_FEATURE_COUNT
    for i in range(len(cf_features)):
        if cf_features[i] == 0.:
            cf_checkbox_list[i] = 'checked'

    # load comments
    txtbox_1, txtbox_2, txtbox_3 = load_textboxes(server_storage_path, global_user_name, disease_name, patient_id, patient_year)


    cf_value1 = cf_values[n]
    cf_value2 = 0.


 
    return render_template('no_hide.html',
        n = n,
        patient_year  = patient_year,
        patient_id   = patient_id,
        patient_info = uncertainty_ranked_patient_info,
        feature_unit = uncertainty_ranked_feature_unit_list,
        patient_label = patient_label,
        disease_name = disease_name,
        disease_offset = num_per_sample,
        checkbox_list = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list=uncertainty_ranked_feature_name_list,
        feature_contrib_list=uncertainty_ranked_feature_contrib_percentage,
        feature_contrib_max=feature_contrib_max,

        current_year_str = current_year_str,
        current_year_act = current_year_act,

        graph_mask = graph_mask,
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )


# 'appp.route' tells what URL should trigger our function
@app.route('/button_year', methods=['GET', 'POST'])
def button_year(): 
    global patient_year
    global patient_id
    global patient_info

    global year_button_idx
    global hide_features
    global stopwatch_start
    global graph_mask

    global cf_features
    global cf_feature_name
    global cf_feature_value
    global cf_value1
    global cf_value2
 

    if request.method == 'POST':
        # recall python url
        n = int(request.form['n'])
        year_button_idx = int(request.form['y'])

        # Bring checked form of checkbox
        hide_idx_list = list(map(lambda x: int(x), request.form.getlist('hide')))

        # Omit [,]
        hide_features = str(hide_idx_list[1:-1])
    else:
        n = int(request.args.get('n'))
        year_button_idx = int(request.args.get('y'))
        hide_idx_list = None


    # call reverse index (sorted by data['InfluenceScores'] or data['PredictionUncertainty'])
    n = min(max(0, n), total_n - 1)
    row = n // num_per_sample
    col = n % num_per_sample

    # Only Infl setting
    data = Infl_data_list[row]
    output = Infl_output[row]

    # small -> large Influence
    n_th_patient = sample_partition - col - 1
    disease_name = disease_list[row]


    patient_id    = n_th_patient
    patient_label = int(data['RealValuesLabels'][n_th_patient][0])

    total_year_exists = 0
    for i in range(len(data['FeatureContribution'][n_th_patient])):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            total_year_exists += 1

    patient_info = preprocess_real_inputs(data['RealValuesInput'][n_th_patient][total_year_exists - year_button_idx - 1]) 

    # call FeatureContribution
    feature_contrib_list = data['FeatureContribution'][n_th_patient][total_year_exists - year_button_idx - 1]
    abs_feature_contrib_list = np.absolute(feature_contrib_list)

    # get rank by uncertainty
    fc_uncertainty = data['FC_Uncertainty'][n_th_patient][total_year_exists - year_button_idx - 1]
    uncertainty_ranks = fc_uncertainty.argsort()[::-1]

    # sort feature_name_list & feature_contrib_list
    np_patient_info         = np.array(patient_info.copy())
    np_feature_unit_list    = np.array(feature_unit_list.copy())
    np_feature_name_list    = np.array(feature_name_list.copy())
    np_feature_contrib_list = np.array(abs_feature_contrib_list.copy())

    
    # Visualize Top-10 Uncertain Feature
    uncertainty_ranked_patient_info = np_patient_info[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_unit_list = np_feature_unit_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_name_list = np_feature_name_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_list = np_feature_contrib_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_sum = sum(uncertainty_ranked_feature_contrib_list)
    uncertainty_ranked_feature_contrib_percentage = [x/uncertainty_ranked_feature_contrib_sum for x in uncertainty_ranked_feature_contrib_list]
    feature_contrib_max = 1.0


    # call Feature is Checked
    checkbox_list = [''] * FEATURE_NUMBER 
    current_output = output[n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][uncertainty_ranks]
    for i in range(len(checkbox_list)):
        if current_output[i] == 0.:
            checkbox_list[i] = 'checked'

    # call CF is Checked
    cf_checkbox_list = [''] * VIS_FEATURE_COUNT
    for i in range(len(cf_features)):
        if cf_features[i] == 0.:
            cf_checkbox_list[i] = 'checked'

    
    # collect train_year of PID
    current_year_str = ['', '', '', '']
    current_year_act = ['disabled', 'disabled', 'disabled', 'disabled']
    old_age = int(data['RealValuesInput'][n_th_patient][total_year_exists-1][1])
    offset = TIME_STEP - total_year_exists
    for i in range(data['FeatureContribution'][n_th_patient].shape[0]):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            current_age = int(data['RealValuesInput'][n_th_patient][i][1])
            current_year_str[offset + i] = 2012 - old_age + current_age
            current_year_act[offset + i] = None


    # load comments
    patient_year = current_year_str[-year_button_idx-1]
    txtbox_1, txtbox_2, txtbox_3 = load_textboxes(server_storage_path, global_user_name, disease_name, patient_id, patient_year)


    cf_value1 = cf_values[n]
    cf_value2 = 0.


    return render_template('no_hide.html',
        n = n,
        patient_year  = patient_year,
        patient_id   = patient_id,
        patient_info = uncertainty_ranked_patient_info,
        feature_unit = uncertainty_ranked_feature_unit_list,
        patient_label = patient_label,
        disease_name = disease_name,
        disease_offset = num_per_sample,
        checkbox_list = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list=uncertainty_ranked_feature_name_list,
        feature_contrib_list=uncertainty_ranked_feature_contrib_percentage,
        feature_contrib_max=feature_contrib_max,

        current_year_str = current_year_str,
        current_year_act = current_year_act,
        
        graph_mask = graph_mask,
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )


# 'appp.route' tells what URL should trigger our function
@app.route('/cf_estimation', methods=['GET', 'POST'])
def cf_estimation():
    global patient_year
    global patient_id
    global patient_info

    global year_button_idx
    global hide_features
    global output
    global graph_mask

    global cf_features
    global cf_feature_name
    global cf_feature_value
    global cf_value1
    global cf_value2


    if request.method == 'POST':
        # recall python url
        n = int(request.form['n'])

        # Bring checked form of checkbox
        hide_idx_list = list(map(lambda x: int(x), request.form.getlist('check')))

        # Omit [, ]
        hide_features = str(hide_idx_list)[1:-1]

        # Read comment 
        # text1 = request.form['text1']
    else:
        n = int(request.args.get('n'))
        hide_idx_list = None
        # text1 = None


    # call reverse index (sorted by data['InfluenceScores'] or data['PredictionUncertainty'])
    n = min(max(0, n), total_n - 1)
    row = n // num_per_sample
    col = n % num_per_sample


    # Only Infl setting
    data = Infl_data_list[row]
    output = Infl_output[row]


    # small -> large Influence
    n_th_patient = sample_partition - col - 1
    disease_name = disease_list[row]
    patient_id   = n_th_patient
    patient_label = int(data['RealValuesLabels'][n_th_patient][0])
    

    total_year_exists = 0
    for i in range(len(data['FeatureContribution'][n_th_patient])):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            total_year_exists += 1

    patient_info = preprocess_real_inputs(data['RealValuesInput'][n_th_patient][total_year_exists - year_button_idx - 1])

    # call FeatureContribution
    feature_contrib_list = data['FeatureContribution'][n_th_patient][total_year_exists - year_button_idx - 1]
    abs_feature_contrib_list = np.absolute(feature_contrib_list)

    # get rank by uncertainty
    fc_uncertainty = data['FC_Uncertainty'][n_th_patient][total_year_exists - year_button_idx - 1]
    uncertainty_ranks = fc_uncertainty.argsort()[::-1]

    # sort feature_name_list & feature_contrib_list
    np_patient_info        = np.array(patient_info.copy())
    np_feature_unit_list    = np.array(feature_unit_list.copy())
    np_feature_name_list    = np.array(feature_name_list.copy())
    np_feature_contrib_list = np.array(abs_feature_contrib_list.copy())


    # Visualize Top-VIS_FEATURE_COUNT Uncertain Feature
    uncertainty_ranked_patient_info = np_patient_info[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_unit_list = np_feature_unit_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_name_list = np_feature_name_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_list = np_feature_contrib_list[uncertainty_ranks].tolist()[:VIS_FEATURE_COUNT]
    uncertainty_ranked_feature_contrib_sum = sum(uncertainty_ranked_feature_contrib_list)
    uncertainty_ranked_feature_contrib_percentage = [x/uncertainty_ranked_feature_contrib_sum for x in uncertainty_ranked_feature_contrib_list]
    feature_contrib_max = 1.0


    # collect train_year of PID
    current_year_str = ['', '', '', '']
    current_year_act = ['disabled', 'disabled', 'disabled', 'disabled']
    old_age = int(data['RealValuesInput'][n_th_patient][total_year_exists-1][1])
    offset = TIME_STEP - total_year_exists
    for i in range(data['FeatureContribution'][n_th_patient].shape[0]):
        if not np.all(data['FeatureContribution'][n_th_patient][i] == 0):
            current_age = int(data['RealValuesInput'][n_th_patient][i][1])
            current_year_str[offset + i] = 2012 - old_age + current_age
            current_year_act[offset + i] = None
   
    patient_year = current_year_str[-year_button_idx-1]

    # PDB_TRACING -> hide_idx_list
    print('\nrequest.method', request.method)
    print('hide_idx_list', hide_idx_list)
    print('')

    # call Feature is Checked
    checkbox_list = [''] * FEATURE_NUMBER
    current_output = output[n_th_patient - (sample_partition - num_per_sample)][total_year_exists - year_button_idx - 1][uncertainty_ranks]
    for i in range(len(checkbox_list)):
        if current_output[i] == 0.:
            checkbox_list[i] = 'checked'

    cf_features[:] = 1.
    for hf in hide_idx_list:
        cf_features[int(hf) - 1] = 0.

    # call CF is Checked
    cf_checkbox_list = [''] * VIS_FEATURE_COUNT
    for i in range(len(cf_features)):
        if cf_features[i] == 0.:
            cf_checkbox_list[i] = 'checked'
            
            # cf_feature_name
            cf_feature_name_list = uncertainty_ranked_feature_name_list[i]
            cf_feature_name      = cf_feature_name_list[0]
            cf_feature_value     = "%.4f" % uncertainty_ranked_feature_contrib_percentage[i]
            for j in range(1, len(cf_feature_name_list)):
                cf_feature_name += " %s" % cf_feature_name_list[j]


    # load comments
    txtbox_1, txtbox_2, txtbox_3 = load_textboxes(server_storage_path, global_user_name, disease_name, patient_id, patient_year)


    ###############################################################################
    # NEED TO MERGE WITH MODEL TO GET ACTUAL ESTIMATION ŷ_cf
    ###############################################################################
    cf_value1 = cf_values[n]
    cf_rand = random.randint(5, 25) / 10.
    cf_value2 = min(1., cf_value1 * cf_rand)
    

    return render_template('no_hide.html',
        n = n,
        patient_year = patient_year,
        patient_id   = patient_id,
        patient_info = uncertainty_ranked_patient_info,
        feature_unit = uncertainty_ranked_feature_unit_list,
        patient_label = patient_label,
        disease_name = disease_name,
        disease_offset = num_per_sample,
        checkbox_list = checkbox_list,
        txtbox_1 = txtbox_1,
        txtbox_2 = txtbox_2,
        txtbox_3 = txtbox_3,

        feature_name_list=uncertainty_ranked_feature_name_list,
        feature_contrib_list=uncertainty_ranked_feature_contrib_percentage,
        feature_contrib_max=feature_contrib_max,

        current_year_str = current_year_str,
        current_year_act = current_year_act,

        graph_mask = graph_mask,
        cf_checkbox_list = cf_checkbox_list,
        cf_feature_name  = cf_feature_name,
        cf_feature_value = cf_feature_value,
        cf_value1        = cf_value1,
        cf_value2        = cf_value2,
    )


 

# shows main page
@app.route('/')
def main():
    return render_template('path.html')

 
if __name__ == "__main__":
    app.run(debug = True)






