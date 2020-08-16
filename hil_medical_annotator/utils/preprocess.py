import os

def preprocess_feature_contrib(inputs):
    outputs = []
    assert(len(inputs) == 34)
    for i in range(34):
        outputs.append(round(inputs[i], 3))
    return outputs


def preprocess_real_inputs(inputs):
    outputs = []

    # check inputs is proper or not
    assert(len(inputs) == 34)
    if inputs[0] == 0:
        outputs.append('여')
    else:
        outputs.append('남')

    outputs.append(int(inputs[1]))
    for i in range(2, 12):
        outputs.append(inputs[i])
    for i in range(12, 19):
        if inputs[i] == 0:
            outputs.append('아니오')
        else:
            outputs.append('예')
    for i in range(19, 29):
        outputs.append(inputs[i])
    for i in range(29, 34):
        if inputs[i] == 0:
            outputs.append('아니오')
        else:
            outputs.append('예')
    return outputs


def load_textboxes(storage_path, user_name, disease_name, patient_id, patient_year):
    txt_result_name_1 = os.path.join(storage_path, user_name, 'Explanations', '%s_%.12d_%d_1.txt' % ('%s_Samples' % disease_name.lower(), patient_id, patient_year))
    txt_result_name_2 = os.path.join(storage_path, user_name, 'Explanations', '%s_%.12d_%d_2.txt' % ('%s_Samples' % disease_name.lower(), patient_id, patient_year))
    txt_result_name_3 = os.path.join(storage_path, user_name, 'Explanations', '%s_%.12d_%d_3.txt' % ('%s_Samples' % disease_name.lower(), patient_id, patient_year))

    txtbox_1 = ''
    txtbox_2 = ''
    txtbox_3 = ''

    if os.path.exists(txt_result_name_1):
        with open(txt_result_name_1, 'rt') as f:
            for line in f:
                txtbox_1 += line
    if os.path.exists(txt_result_name_2):
        with open(txt_result_name_2, 'rt') as f:
            for line in f:
                txtbox_2 += line
    if os.path.exists(txt_result_name_3):
        with open(txt_result_name_3, 'rt') as f:
            for line in f:
                txtbox_3 += line

    return txtbox_1, txtbox_2, txtbox_3
