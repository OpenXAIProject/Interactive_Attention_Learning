import numpy as np
import os
import time
#import IPython
from scipy.stats import pearsonr
import pdb
import pickle
import random

def get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test):
    def try_check(idx_to_check, label):
        Y_train_fixed = np.copy(Y_train_flipped)
        Y_train_fixed[idx_to_check] = Y_train[idx_to_check]
        model.update_train_x_y(X_train, Y_train_fixed)
        model.train()
        check_num = np.sum(Y_train_fixed != Y_train_flipped)
        check_loss, check_acc = model.sess.run(
            [model.loss_no_reg, model.accuracy_op], 
            feed_dict=model.all_test_feed_dict)
        print('%20s: fixed %3s labels. Loss %.5f. Accuracy %.3f.' % (
              label, check_num, check_loss, check_acc))
        return check_num, check_loss, check_acc
    return try_check


def Find_influential_training_input(model, 
                                    iter_to_load, 
                                    force_refresh,
                                    num_to_choose_test, 
                                    num_to_choose_train, 
                                    num_steps, 
                                    random_seed,
                                    remove_type,
                                    model_name,
                                    approx_type,
                                    loss_type,
                                    test_description,
                                    train_dir,
                                    num_sampling,
                                    task,
                                    stage,
                                    ):

    np.random.seed(random_seed)
    model.load_checkpoint(iter_to_load)
    sess = model.sess

    # #Biggest gap
    chosen_test_indices = model.Collect_incorrect_test_indices(num_to_choose=num_to_choose_test)

    y_test = model.data_sets.test.labels[chosen_test_indices]
    print('Number of Test idices: %s' % len(chosen_test_indices))
    print('Test label: %s' % y_test)

    #Indices with ighest influence
    predicted_loss_diffs = model.get_influence_on_test_loss(
                                                            [chosen_test_indices],
                                                            np.arange(len(model.data_sets.train.labels)),
                                                           )

    chosen_training_indices = np.argsort(np.abs(predicted_loss_diffs))[-num_to_choose_train:]

    filename = os.path.join(train_dir, '%s-chosen_training_indices.npz' % (model_name))
    np.savez(filename, chosen_training_indices=chosen_training_indices)
    
    chosen_training_influence_scores = predicted_loss_diffs[chosen_training_indices]
    print('Number of influence_scores after collecting: %s' % chosen_training_influence_scores.shape[0])

    np.save('chosen_training_indices.npy', chosen_training_indices)
    print('Number of influence_scores after collecting: %s' % chosen_training_influence_scores.shape[0])

    #Sampling feature contribution
    f_samples = np.zeros((num_sampling, \
                        chosen_training_influence_scores.shape[0], \
                        model.data_sets.train.x.shape[1], \
                        model.data_sets.train.x.shape[2], \
                      ))
    for i_ in range(num_sampling):
        featureCont, RealValues_Input, RealValues_Labels = model.Feature_contributions_for_chosen_instance(chosen_training_indices,\
                                                           type_instance='train')
        f_samples[i_,:,:,:] = featureCont

    FeatureContribution = np.mean(f_samples, axis=0)
    FC_Uncertainty = np.var(f_samples, axis=0)


    perturbed_input = model.data_sets.train.x[chosen_training_indices]
    perturbed_labels = model.data_sets.train.labels[chosen_training_indices]
    feature_if_dif = np.zeros([perturbed_input.shape[0], perturbed_input.shape[1], perturbed_input.shape[2]])
    
    def random_floats(low, high, size):
        return [random.uniform(low, high) for _ in range(size)]

    mean_feature_aggregator=[]
    std_feature_aggregator=[]
    for feature_ in range(perturbed_input.shape[2]):
        mean_feature_tmp=[]
        std_feature_tmp=[]
        for instance_ in range(perturbed_input.shape[0]):
            for step_ in range(perturbed_input.shape[1]):
                mean_feature_tmp.append(perturbed_input[instance_][step_][feature_])
                std_feature_tmp.append(perturbed_input[instance_][step_][feature_])

        mean_feature_aggregator.append(np.mean(mean_feature_tmp))
        std_feature_aggregator.append(np.std(std_feature_tmp))


        delta = random_floats(mean_feature_aggregator[feature_]-(2*std_feature_aggregator[feature_]), 
                              mean_feature_aggregator[feature_]+(2*std_feature_aggregator[feature_]),
                              1
                              )

        for instance_ in range(perturbed_input.shape[0]):
            for step_ in range(perturbed_input.shape[1]):
                original_value = perturbed_input[instance_, step_, feature_]
                perturbed_input[instance_][step_][feature_] = perturbed_input[instance_][step_][feature_] + delta

        all_feature_predicted_loss=[]
        for i_ in range(10):
            tmp_feature_predicted_loss = model.get_feature_influence_on_test_loss(
                                                                              [chosen_test_indices],
                                                                              [chosen_training_indices],
                                                                              perturbed_input,
                                                                              perturbed_labels,
                                                                              force_refresh=False,
                                                                                 )
            all_feature_predicted_loss.append(tmp_feature_predicted_loss)

        feature_predicted_loss = np.mean(all_feature_predicted_loss)
        tmp_feature_if = np.abs(chosen_training_influence_scores - feature_predicted_loss)
        
        for timestep_ in range(feature_if_dif.shape[1]):
            feature_if_dif[:, timestep_, feature_] = tmp_feature_if

    samples = {
                "FeatureContribution": FeatureContribution,
                "FC_Uncertainty": FC_Uncertainty,
                "Feature_Influence_Score": feature_if_dif,
                "RealValuesInput": RealValues_Input,
                "RealValuesLabels": RealValues_Labels,
                "InfluenceScores": chosen_training_influence_scores,
                "ChosenTrainingIndices": chosen_training_indices,
               }

    result_folder = os.path.join('/IAL_cerebral1_variants/ifif_Samples', task)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(os.path.join(result_folder, stage+task+'_Samples_ifif.pickle'), 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('IF_Training file has been created.')
    return samples



    #Sampling feature contribution
    samples = np.zeros((num_sampling, \
                        chosen_training_influence_scores.shape[0], \
                        model.data_sets.train.x.shape[1], \
                        model.data_sets.train.x.shape[2], \
                      ))
    for i_ in range(num_sampling):
        featureCont, RealValues_Input, RealValues_Labels = model.Feature_contributions_for_chosen_instance(chosen_training_indices,\
                                                           type_instance='train')
        samples[i_,:,:,:] = featureCont

    FeatureContribution = np.mean(samples, axis=0)
    FC_Uncertainty = np.var(samples, axis=0)
    
    samples = {
                "FeatureContribution": FeatureContribution,
                "FC_Uncertainty": FC_Uncertainty,
                "RealValuesInput": RealValues_Input,
                "RealValuesLabels": RealValues_Labels,
                "InfluenceScores": chosen_training_influence_scores,
                "ChosenTrainingIndices": chosen_training_indices,
               }

    result_folder = os.path.join('/IAL_cerebral1/Samples', task)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(os.path.join(result_folder, stage+task+'_Samples.pickle'), 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('IF_Training file has been created.')
    return samples


def further_stage_Find_influential_training_input(model, 
                                                  iter_to_load, 
                                                  force_refresh,
                                                  num_to_choose_test, 
                                                  num_to_choose_train, 
                                                  num_steps, 
                                                  random_seed,
                                                  remove_type,
                                                  model_name,
                                                  approx_type,
                                                  loss_type,
                                                  test_description,
                                                  train_dir,
                                                  num_sampling,
                                                  task,
                                                  stage,
                                                 ):

    np.random.seed(random_seed)
    model.retrain_load_checkpoint(iter_to_load)
    sess = model.sess

    #Biggest gap
    chosen_test_indices = model.Collect_incorrect_test_indices(num_to_choose=num_to_choose_test)

    y_test = model.data_sets.test.labels[chosen_test_indices]
    print('Number of Test idices: %s' % len(chosen_test_indices))
    print('Test label: %s' % y_test)

    #Indices with ighest influence
    start_time = time.time()
    predicted_loss_diffs = model.get_influence_on_test_loss(
                                                            [chosen_test_indices],
                                                            np.arange(len(model.data_sets.train.labels)),
                                                           )
    duration = time.time() - start_time


    chosen_training_indices = np.argsort(np.abs(predicted_loss_diffs))[-num_to_choose_train:]

    filename = os.path.join(train_dir, '%s-chosen_training_indices.npz' % (model_name))
    np.savez(filename, chosen_training_indices=chosen_training_indices)
    
    chosen_training_influence_scores = predicted_loss_diffs[chosen_training_indices]
    print('Number of influence_scores after collecting: %s' % chosen_training_influence_scores.shape[0])

    #Sampling feature contribution
    samples = np.zeros((num_sampling, \
                        chosen_training_influence_scores.shape[0], \
                        model.data_sets.train.x.shape[1], \
                        model.data_sets.train.x.shape[2], \
                      ))
    for i_ in range(num_sampling):
        featureCont, RealValues_Input, RealValues_Labels = model.Feature_contributions_for_chosen_instance(chosen_training_indices,\
                                                           type_instance='train')
        samples[i_,:,:,:] = featureCont

    FeatureContribution = np.mean(samples, axis=0)
    FC_Uncertainty = np.var(samples, axis=0)
    
    samples = {
                "FeatureContribution": FeatureContribution,
                "FC_Uncertainty": FC_Uncertainty,
                "RealValuesInput": RealValues_Input,
                "RealValuesLabels": RealValues_Labels,
                "InfluenceScores": chosen_training_influence_scores,
                "ChosenTrainingIndices": chosen_training_indices,
               }

    result_folder = 'IAL_%s/Samples' % task
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    logfile = os.path.join(result_folder, '%s%s_log.txt' % (stage,task))
    with open(logfile, 'wt') as txtfile:
        txtfile.write('%s\n' % duration)

    with open(os.path.join(result_folder, stage+task+'_Samples.pickle'), 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('IF_Training file has been created.')
    return samples


def Find_final_evaluation_input(model,
                                iter_to_load,
                                force_refresh,
                                num_to_choose_test,
                                num_to_choose_train,
                                num_steps,
                                random_seed,
                                remove_type,
                                model_name,
                                approx_type,
                                loss_type,
                                test_description,
                                train_dir,
                                num_sampling,
                                task,
                                stage,
                                ):

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    chosen_final_evaluation_indices = np.arange(num_to_choose_train)
    chosen_final_evaluation_influence_scores = np.zeros(len(chosen_final_evaluation_indices), dtype=np.float32)
    print('Number of influence_scores after collecting: %s' % chosen_final_evaluation_influence_scores.shape[0])

    #Sampling feature contribution
    samples = np.zeros((num_sampling, \
                        chosen_final_evaluation_influence_scores.shape[0], \
                        model.data_sets.train.x.shape[1], \
                        model.data_sets.train.x.shape[2], \
                      ))

    for i_ in range(num_sampling):
        featureCont, RealValues_Input, RealValues_Labels = model.Feature_contributions_for_chosen_instance(chosen_final_evaluation_indices,\
                                                           type_instance='final_evaluation')
        samples[i_,:,:,:] = featureCont

    FeatureContribution = np.mean(samples, axis=0)
    FC_Uncertainty = np.var(samples, axis=0)

    samples = {
                "FeatureContribution": FeatureContribution,
                "FC_Uncertainty": FC_Uncertainty,
                "RealValuesInput": RealValues_Input,
                "RealValuesLabels": RealValues_Labels,
                "InfluenceScores": chosen_final_evaluation_influence_scores,
                "ChosenTrainingIndices": chosen_final_evaluation_indices,
               }


    result_folder = os.path.join('/EHR_final_evaluation', task)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    with open(os.path.join(result_folder, stage+'final_evaluation_'+task+'_Samples.pickle'), 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('IF_Training file has been created.')
    return samples
