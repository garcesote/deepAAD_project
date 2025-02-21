import torch
from utils.functional import set_seeds, get_mdl_name, verbose, get_filename, get_subjects, get_data_path
import numpy as np
import os
import pickle
from utils.datasets import CustomDataset
import json
import argparse
import wandb
import yaml
import sys

def main(config, dataset: str, key: str, cross_val: bool, wandb_upload: bool):

    # Data path parameters
    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'euroacustics'
    exp_name = config['exp_name']

    # REPRODUCIBILITY
    if 'seed' in config.keys(): 
        set_seeds(config['seed'])
        exp_name =  exp_name + '_' + config['seed']
    else: 
        set_seeds() # default seed = 42

    for run in config['runs']:

        # Load all the parameters
        linear_model = run['linear_model']
        dataset_params = run['dataset_params']
        preproc_mode = run.get('train_params').get('preproc_mode')

        # Config dataset
        dataset_params['leave_one_out'] = True if key == 'subj_independent' else False
        data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)

        # window_list = [64, 128, 320, 640, 1280, 2560]
        window_list = [64, 128, 320, 640, 1600, 3200]

        time_shift = 100 # Null distribution
        
        # Add extensions to the model name depending on the params
        model_name = get_mdl_name(run, linear=True)

        # Cross validation 
        if cross_val:
            n_folds = 5
        else:
            n_folds = 1
        
        for cv_fold in range(n_folds):

            if not cross_val: cv_fold = None

            # WANDB INIT
            if cross_val: run['cv_fold'] = cv_fold
            run['key'] = key
            run['dataset'] = dataset
            if wandb_upload: wandb.init(project=project, name=exp_name, tags=['evaluation_linear'], config=run)

            window_accuracies = {win//64: None for win in window_list}

            # DEFINE THE SAVE PATH
            cv_folder = f'cvFold={cv_fold}' if cross_val else 'global_validation'
            dst_save_path = os.path.join(global_path, 'results', project, key, 'eval_metrics', dataset+'_data', model_name, cv_folder)
            decAcc_save_path = os.path.join(global_path, 'results', project, key, 'decode_accuracy', dataset+'_data', model_name, cv_folder)

            for window in window_list:

                str_win = str(window//64)+'s'

                eval_results = {}
                nd_results = {} # construct a null distribution when evaluating
                dec_results = []
                eval_mean_results = []

                selected_subjects = get_subjects(dataset)

                for subj in selected_subjects:

                    verbose('evaluate', key, subj, dataset, linear_model, window=str_win, cv_fold=cv_fold)

                    # GET THE MODEL PATH
                    mdl_load_path = os.path.join(global_path, 'results', project, key, 'models', dataset+'_data', model_name)
                    if key != 'population':
                        mdl_load_path = os.path.join(mdl_load_path, subj)
                    
                    mdl_filename = get_filename(mdl_load_path, cv_fold)
                    mdl_load_path = os.path.join(mdl_load_path, mdl_filename)

                    # LOAD THE MODEL
                    mdl = pickle.load(open(mdl_load_path, 'rb'))

                    # LOAD THE DATA
                    test_set = CustomDataset(dataset, data_path, 'test', subj, cv_fold=cv_fold, **dataset_params)
                    test_eeg, test_stima, test_stimb = test_set.eeg, test_set.stima, test_set.stimb
                    test_stim_nd = torch.roll(test_stima.clone().detach(), time_shift)

                    if "Ridge" in linear_model:

                        # EVALUATE WITH THE BEST ALPHA OBTAINED
                        scores_a = mdl.score_in_batches(test_eeg.T, test_stima.T, batch_size=window)
                        scores_nd = mdl.score_in_batches(test_eeg.T, test_stim_nd.T, batch_size=window) # ya selecciona el best alpha solo

                        # DECODING ACCURACY
                        att_corr = 0
                        if test_stimb is not None:
                            scores_b = mdl.score_in_batches(test_eeg.T, test_stimb.T, batch_size=window)
                            for score_a, score_b in zip(scores_a, scores_b):
                                if score_a > score_b:
                                    att_corr += 1

                        accuracy = (att_corr / len(scores_a)) * 100

                    elif linear_model == "CCA":
                        
                        # COMPUTE THE MEAN OF CANONICAL COMPONENTS SCORES
                        scores_a = np.mean(mdl.score_in_batches(test_eeg, test_stima, batch_size=window), axis=1)
                        scores_nd = np.mean(mdl.score_in_batches(test_eeg, test_stim_nd, batch_size=window), axis=1)

                        # GET THE ACCURACY USING THE LDA CLASSIFIER
                        if test_stimb is not None:
                            accuracy = mdl.classify_in_batches(test_eeg, test_stima, test_stimb, batch_size = window)

                    else: raise ValueError('Introduce a valid linear model name')

                    eval_results[subj] = [score for score in np.squeeze(scores_a)]
                    nd_results[subj] = [score for score in np.squeeze(scores_nd)]
                    
                    dec_results.append(accuracy)
                    eval_mean_results.append(np.mean(scores_a))

                    print(f'Subject {subj} | corr_mean {np.mean(scores_a):.4f} | decode_accuracy {accuracy}')

                if wandb_upload:
                    wandb.log({'window': window, 'corr_subj_mean': np.mean(eval_mean_results), 'corr_subj_std': np.std(eval_mean_results), 'decAcc_subj_mean': np.mean(dec_results), 'decAcc_subj_std': np.std(dec_results)})

                # Save the window results to compute mesd
                window_accuracies[window//64] = dec_results

                # SAVE RESULTS
                if not os.path.exists(dst_save_path):
                    os.makedirs(dst_save_path)
                filename = str_win+'_Results'
                json.dump(eval_results, open(os.path.join(dst_save_path, filename),'w'))
                filename = str_win+'_nd_Results'
                json.dump(nd_results, open(os.path.join(dst_save_path, filename),'w'))

                # SAVE ACCURACY RESULTS
                if not os.path.exists(decAcc_save_path):
                    os.makedirs(decAcc_save_path)
                filename = str_win+'_accuracies'
                json.dump(dec_results, open(os.path.join(decAcc_save_path, filename),'w'))
            
            # COMPUTE MESD (default values)
            print('Computing MESD')
            mesd_dict = {'mesd': [], 'N_mesd': [], 'tau_mesd': [], 'p_mesd': []}
            for subj in range(len(selected_subjects)):
                tau = np.array(list(window_accuracies.keys()))
                p = np.array([results[subj] / 100 for results in window_accuracies.values()])
                mesd, N_mesd, tau_mesd, p_mesd = compute_MESD(tau,p)
                print(f"For subject {subj} the minimal expected switch duration is MESD = {mesd} \nat an otpimal working point of (tau, p) = ({tau_mesd}, {p_mesd}) \nwith N = {N_mesd} states in the Markov chain.")
                mesd_results = [mesd, N_mesd, tau_mesd, p_mesd]
                for idx, value in zip(list(mesd_dict.keys()), mesd_results):
                    mesd_dict[idx].append(value)
                # Wandb upload with the specific info of the specific subject
                upload_data = {idx: value[subj] for idx, value in mesd_dict.items()}
                upload_data['subject'] = subj
                if wandb_upload: wandb.log(upload_data)
            if wandb_upload: wandb.log({'mesd_mean': np.mean(mesd_dict['mesd']), 'mesd_median': np.median(mesd_dict['mesd'])})
            json.dump(mesd_dict, open(os.path.join(decAcc_save_path, 'mesd'),'w'))

            if wandb_upload: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval linear script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--config", type=str, default='configs/euroacustics/linear_models.yaml')
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_specific', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--cross_val", action='store_true', help="When included select the cross validation models")        
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    
    args = parser.parse_args()

    wandb_upload = args.wandb

    # Introduce path to the mesd-toolbox
    sys.path.append(r"C:\Users\jaulab\Desktop\AAD\mesd-toolbox\mesd-toolbox-python")
    from mesd_toolbox import compute_MESD
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    with open(args.config, 'r') as archivo:
        config = yaml.safe_load(archivo)
    
    # Llamar a la función de entrenamiento con los argumentos
    main(config, args.dataset, args.key, args.cross_val, wandb_upload)