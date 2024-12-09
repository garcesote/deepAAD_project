import torch
from models.ridge import Ridge
from models.cca import CCA_AAD
import numpy as np
import os
import pickle
from utils.functional import get_data_path, get_trials_len, get_subjects, str2bool
from utils.datasets import CustomDataset
import argparse
import yaml

def main(
        config,
        dataset: str,
        key: str,
    ):

    # Data path parameters
    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'spatial_audio'
    mdl_save_path = os.path.join(global_path, 'results', project, key, 'models')

    for n, run in enumerate(config['runs']):

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

        leave_one_out = True if key == 'subj_independent' else False # Attention! This parameter change the whole sim. (read doc)
        fixed = fixed
        rnd_trials = rnd_trials
        trial_len = get_trials_len(dataset)
        data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)

        if key == 'population':
            selected_subj = [get_subjects(dataset)]
        else:
            selected_subj = get_subjects(dataset)

        for subj in selected_subj:
            
            if key == 'population':
                print(f'Training {linear_model} on all subjects with {dataset} data...')
            elif key == 'subj_independent':
                print(f'Training {linear_model} leaving out {subj} with {dataset} data...')
            else:
                print(f'Training {linear_model} on {subj} with {dataset} data...')

            mdl_prefix = key if key == 'population' else subj

            # LOAD THE DATA
            train_set = CustomDataset(dataset, data_path, 'train', subj, window=trial_len, hop=trial_len, data_type=data_type, 
                                        leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, hrtf=hrtf, eeg_band=eeg_band)
            val_set = CustomDataset(dataset, data_path, 'val',  subj, window=trial_len, hop=trial_len, data_type=data_type, 
                                    leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, hrtf=hrtf, eeg_band=eeg_band)

            if linear_model == 'Ridge':

                alphas = np.logspace(-7,7, 15)
                model_params['alpha'] = alphas
                model_params['trial_len'] = trial_len
                mdl = Ridge(**model_params)

                # TRAIN MODEL
                mdl.fit(train_set.eeg.T, train_set.stima.T)
                
                # VALIDATE AND SELECT BEST ALPHA
                scores = mdl.model_selection(val_set.eeg.T, val_set.stima.T)
                best_alpha = mdl.best_alpha_idx

                model_name = f"{linear_model}_start={model_params['start_lag']}_end={model_params['end_lag']}"
                mdl_filename = f"{mdl_prefix}_alpha={best_alpha}_acc={scores[best_alpha]:.3f}"

                print(f'Ridge trained for {dataset} data with a score of {scores[best_alpha]} with alpha = {best_alpha}')

            elif linear_model == 'CCA':
                
                model_params['trial_len'] = trial_len
                mdl = CCA_AAD(**model_params)
                
                # TRAIN THE CCA PROJECTION MODEL
                mdl.fit_CCA(eeg=train_set.eeg, stim=train_set.stima)

                # TRAIN THE LDA CLASSIFIER MODEL
                mdl.fit_LDA(eeg=train_set.eeg, stima=train_set.stima, stimb=train_set.stimb, batch_size=trial_len)
                
                # VALIDATE THE MODEL
                scores_a = mdl.score_in_batches(val_set.eeg, val_set.stima, batch_size=trial_len)
                scores_b = mdl.score_in_batches(val_set.eeg, val_set.stimb, batch_size=trial_len)
                accuracy = mdl.classify_in_batches(val_set.eeg, val_set.stima, val_set.stimb, batch_size=trial_len)
                
                model_name = f"{linear_model}_enc={model_params['encoder_len']}_dec={model_params['decoder_len']}_comp={model_params['n_components']}"
                if 'max_iter' in model_params.keys(): model_name += ('_max_iter=' + str(model_params['max_iter']))
                if 'tol' in model_params.keys(): model_name += ('_tol=' + str(model_params['tol']))
                mdl_filename = f"{mdl_prefix}_score={np.mean(scores_a):.3f}_acc={accuracy:.3f}"

                print(f'CCA stats on {dataset} dataset: score_a {np.mean(scores_a)} | score_b {np.mean(scores_b)} | val_accuracy {accuracy}%')

            else: raise ValueError('Introduce a valid lineal model between Ridge or CCA')

            # Add extensions to the model name depending on the dataset params
            if preproc_mode is not None: model_name += '_' + preproc_mode
            if eeg_band is not None: model_name += '_' + eeg_band
            if rnd_trials: model_name += '_rnd'
            if hrtf: model_name += '_hrtf'

            dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset
            
            # SAVE THE MODEL
            mdl_folder = os.path.join(mdl_save_path, dataset_filename + '_data', model_name)
            if not os.path.exists(mdl_folder):
                os.makedirs(mdl_folder)
            save_path = os.path.join(mdl_folder, mdl_filename)
            pickle.dump(mdl, open(save_path, "wb"))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 8
    torch.set_num_threads(n_threads)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = "8"  # Número de hilos para OpenMP
    os.environ["MKL_NUM_THREADS"] = "8"  # Número de hilos para MKL (usado por NumPy y SciPy)
    os.environ["NUMEXPR_NUM_THREADS"] = "8"  # Número de hilos para NumExpr
    os.environ["OPENBLAS_NUM_THREADS"] = "8"  # Número de hilos para OpenBLAS
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Para bibliotecas vecLib en macOS
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--config", type=str, default='configs/spatial_audio/linear_models.yaml')
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_specific', help="Key from subj_specific, subj_independent and population")
    
    args = parser.parse_args()

    # Load the yaml file
    with open(args.config, 'r') as archivo:
        config = yaml.safe_load(archivo)

    # Llamar a la función de entrenamiento con los argumentos
    main(
        config,
        args.dataset,
        args.key,
    )