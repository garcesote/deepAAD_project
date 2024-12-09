import torch
from utils.functional import get_filename, get_subjects, str2bool, get_data_path
import numpy as np
import os
import pickle
from utils.datasets import CustomDataset
import json
import argparse
import wandb
import yaml
import sys

def main(
        config,
        dataset: str,
        key: str,
        wandb_upload: bool,
    ):

    # Data path parameters
    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'spatial_audio'

    for run in config['runs']:

        # Load all the parameters
        linear_model = run['linear_model']
        model_params = run['model_params']
        dataset_params = run['dataset_params']
        fixed = dataset_params['fixed']
        rnd_trials = dataset_params['rnd_trials']
        hrtf = dataset_params['hrtf'] if 'hrtf' in dataset_params.keys() else False
        preproc_mode = dataset_params['preproc_mode'] if 'preproc_mode' in dataset_params.keys() else None
        data_type = dataset_params['data_type'] if 'data_type' in dataset_params.keys() else 'mat'
        eeg_band = dataset_params['eeg_band'] if 'eeg_band' in dataset_params.keys() else None
    
        data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)

        window_list = [64, 128, 320, 640, 1280, 2560]
        # window_list = [64, 2560]
        time_shift = 100 # Null distribution
        leave_one_out = True if key == 'subj_independent' else False # Attention! This parameter change the whole sim. (read doc)
        dec_acc = True if dataset != 'skl' else False # skl dataset without unattended stim => dec-acc is not possible
        
        if fixed: assert(dataset == 'jaulab')
        dataset_name = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

        # Add extensions to the model name depending on the params
        if linear_model == 'Ridge':
            model = f"{linear_model}_start={model_params['start_lag']}_end={model_params['end_lag']}"
        elif linear_model == 'CCA':
            model = f"{linear_model}_enc={model_params['encoder_len']}_dec={model_params['decoder_len']}_comp={model_params['n_components']}"
            if 'max_iter' in model_params.keys(): model += ('_max_iter=' + str(model_params['max_iter']))
            if 'tol' in model_params.keys(): model += ('_tol=' + str(model_params['tol']))
        else: raise ValueError("Introduce a valid linear model name between 'Ridge' and 'CCA'")

        if preproc_mode is not None: model += '_' + preproc_mode
        if eeg_band is not None: model += '_' + eeg_band
        if rnd_trials: model += '_rnd'
        if hrtf: model += '_hrtf'

        # DEFINE THE SAVE PATH
        dst_save_path = os.path.join(global_path, 'results', project, key, 'eval_metrics', dataset_name+'_data', model)
        decAcc_save_path = os.path.join(global_path, 'results', project, key, 'decode_accuracy', dataset_name+'_data', model)
        mdl_load_path = os.path.join(global_path, 'results', project, key, 'models', dataset+'_data', model)
        
        exp_name = ('_').join([key, dataset_name, model])
        if wandb_upload: wandb.init(project=project, name=exp_name, tags=['evaluation_ridge'], config=run)

        window_accuracies = {win//64: None for win in window_list}
        
        for window in window_list:

            block_size = window
            dataset = dataset

            eval_results = {}
            nd_results = {} # construct a null distribution when evaluating
            dec_results = []
            eval_mean_results = []

            selected_subjects = get_subjects(dataset)

            for subj in selected_subjects:

                print(f'Evaluating {linear_model} on window {window//64}s with {dataset_name} dataset for subj {subj}')

                # GET THE MODEL PATH
                if key == 'population':
                    mdl_filename = os.listdir(mdl_load_path)[0]
                else:
                    mdl_filename = get_filename(mdl_load_path, subj)

                # LOAD THE MODEL
                mdl = pickle.load(open(os.path.join(mdl_load_path, mdl_filename), 'rb'))

                # LOAD THE DATA
                test_set = CustomDataset(dataset, data_path, 'test', subj, window=block_size, hop=block_size, data_type=data_type,
                                        leave_one_out=leave_one_out, fixed=fixed, rnd_trials=rnd_trials, hrtf=hrtf, eeg_band = eeg_band)
                test_eeg, test_stima, test_stimb = test_set.eeg, test_set.stima, test_set.stimb
                test_stim_nd = torch.roll(test_stima.clone().detach(), time_shift)

                if linear_model == "Ridge":

                    # EVALUATE WITH THE BEST ALPHA OBTAINED
                    scores_a = mdl.score_in_batches(test_eeg.T, test_stima.T, batch_size=block_size)
                    scores_nd = mdl.score_in_batches(test_eeg.T, test_stim_nd.T, batch_size=block_size) # ya selecciona el best alpha solo

                    # DECODING ACCURACY
                    att_corr = 0
                    if dec_acc:
                        scores_b = mdl.score_in_batches(test_eeg.T, test_stimb.T, batch_size=block_size)
                        for score_a, score_b in zip(scores_a, scores_b):
                            if score_a > score_b:
                                att_corr += 1

                    accuracy = (att_corr / len(scores_a)) * 100

                elif linear_model == "CCA":
                    
                    # COMPUTE THE MEAN OF CANONICAL COMPONENTS SCORES
                    scores_a = np.mean(mdl.score_in_batches(test_eeg, test_stima, batch_size=block_size), axis=1)
                    scores_nd = np.mean(mdl.score_in_batches(test_eeg, test_stim_nd, batch_size=block_size), axis=1)

                    # GET THE ACCURACY USING THE LDA CLASSIFIER
                    if dec_acc:
                        accuracy = mdl.classify_in_batches(test_eeg, test_stima, test_stimb, batch_size = block_size)

                eval_results[subj] = [score for score in np.squeeze(scores_a)]
                nd_results[subj] = [score for score in np.squeeze(scores_nd)]
                
                dec_results.append(accuracy)
                eval_mean_results.append(np.mean(scores_a))

                print(f'Subject {subj} | corr_mean {np.mean(scores_a):.4f} | decode_accuracy {accuracy}')

            if wandb_upload:
                    wandb.log({'window': block_size, 'corr_subj_mean': np.mean(eval_mean_results), 'corr_subj_std': np.std(eval_mean_results), 'decAcc_subj_mean': np.mean(dec_results), 'decAcc_subj_std': np.std(dec_results)})

            # Save the window results to compute mesd
            window_accuracies[window//64] = dec_results

            str_win = str(block_size//64)+'s'

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

    parser = argparse.ArgumentParser(description="Eval Ridge script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--config", type=str, default='configs/spatial_audio/eval_mesd_linear.yaml')
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_specific', help="Key from subj_specific, subj_independent and population")
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
    main(
        config,
        args.dataset,
        args.key,
        wandb_upload
    )