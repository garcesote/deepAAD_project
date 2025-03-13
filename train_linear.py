import torch
from models.ridge import Ridge, Ridge_SKL
from models.cca import CCA_AAD
import numpy as np
import os
import pickle
from utils.functional import get_mdl_name, verbose, set_seeds, get_data_path, get_trials_len, get_subjects
from utils.datasets import CustomDataset
import argparse
import yaml

def main(config, dataset: str, key: str, cross_val: bool):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'euroacustics'
    mdl_save_path = os.path.join(global_path, 'results', project, key, 'models')
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
        model_params = run['model_params']
        dataset_params = run['dataset_params']
        preproc_mode = run.get('train_params').get('preproc_mode')

        # Config dataset
        dataset_params['leave_one_out'] = True if key == 'subj_independent' else False
        trial_len = get_trials_len(dataset)
        data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)

        if key == 'population':
            selected_subj = [get_subjects(dataset)]
        else:
            selected_subj = get_subjects(dataset)

        # Cross validation 
        if cross_val:
            n_folds = 5
        else:
            n_folds = 1

        for cv_fold in range(n_folds):

            if not cross_val: cv_fold = None

            for subj in selected_subj:
                
                verbose('train', key, subj, dataset, linear_model)

                mdl_prefix = linear_model if not cross_val else f'{linear_model}_cvFold={cv_fold}'

                # LOAD THE DATA
                train_set = CustomDataset(dataset, data_path, 'train', subj, cv_fold=cv_fold, **dataset_params)
                val_set = CustomDataset(dataset, data_path, 'val',  subj, cv_fold=cv_fold, **dataset_params)

                if 'Ridge' in linear_model:

                    alphas = np.logspace(-7,7, 15)
                    model_params['alpha'] = alphas
                    model_params['trial_len'] = trial_len
                    
                    if linear_model == 'Ridge_scikit':
                        mdl = Ridge_SKL(**model_params)

                        # TRAIN MODEL
                        mdl.train(X_train=train_set.eeg.T, y_train=train_set.stima.T,
                                  X_val=val_set.eeg.T, y_val=val_set.stima.T)
                        best_alpha = mdl.best_alpha
                        best_score = mdl.best_corr

                    else:
                        mdl = Ridge(**model_params)

                        # TRAIN MODEL
                        mdl.fit(train_set.eeg.T, train_set.stima.T)
                        
                        # VALIDATE AND SELECT BEST ALPHA
                        scores = mdl.model_selection(val_set.eeg.T, val_set.stima.T)
                        best_alpha = alphas[mdl.best_alpha_idx]
                        best_score = scores[mdl.best_alpha_idx]

                    mdl_filename = f"{mdl_prefix}_alpha={best_alpha:.2e}_acc={best_score:.3f}"
                    
                    print(f'Ridge trained for {dataset} data with a score of {best_score:.3f} with alpha = {best_alpha:.2e}')

                elif linear_model == 'CCA':
                    
                    model_params['trial_len'] = trial_len
                    mdl = CCA_AAD(**model_params)
                    
                    # TRAIN THE CCA PROJECTOR MODEL
                    mdl.fit_CCA(eeg=train_set.eeg, stim=train_set.stima)

                    # TRAIN THE LDA CLASSIFIER MODEL: val set to select the optimal components that maximize the accuracy
                    mdl.fit_LDA(eeg=train_set.eeg, stima=train_set.stima, stimb=train_set.stimb, 
                                eeg_val=val_set.eeg, stima_val=val_set.stima, stimb_val=val_set.stimb, 
                                batch_size=trial_len//10)
                    
                    scores_a = mdl.score_in_batches(train_set.eeg, train_set.stima, batch_size=trial_len)
                    scores_b = mdl.score_in_batches(train_set.eeg, train_set.stimb, batch_size=trial_len)

                    mdl_filename = f"{mdl_prefix}_score={np.mean(scores_a):.3f}_acc={mdl.best_accuracy:.3f}_nComponents={mdl.n_components}"

                    print(f'CCA stats on {dataset} dataset: n_components: {mdl.n_components} | val_accuracy: {mdl.best_accuracy}% | score_a: {np.mean(scores_a)} | score_b: {np.mean(scores_b)} | val_accuracy: {mdl.best_accuracy}%')

                else: raise ValueError('Introduce a valid lineal model between Ridge or CCA')

                model_name = get_mdl_name(run, linear=True)
                
                # SAVE THE MODEL
                mdl_folder = os.path.join(mdl_save_path, dataset + '_data', model_name, subj)
                if not os.path.exists(mdl_folder):
                    os.makedirs(mdl_folder)
                save_path = os.path.join(mdl_folder, mdl_filename)
                pickle.dump(mdl, open(save_path, "wb"))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    # n_threads = 8
    # torch.set_num_threads(n_threads)
    # os.environ['OMP_NUM_THREADS'] = str(n_threads)
    # os.environ["OMP_NUM_THREADS"] = "8"  # Número de hilos para OpenMP
    # os.environ["MKL_NUM_THREADS"] = "8"  # Número de hilos para MKL (usado por NumPy y SciPy)
    # os.environ["NUMEXPR_NUM_THREADS"] = "8"  # Número de hilos para NumExpr
    # os.environ["OPENBLAS_NUM_THREADS"] = "8"  # Número de hilos para OpenBLAS
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Para bibliotecas vecLib en macOS
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--config", type=str, default='configs/euroacustics/cca_search.yaml')
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--cross_val", action='store_true', help="When included perform a 5 cross validation for the train_set")
    
    args = parser.parse_args()

    # Load the yaml file
    with open(args.config, 'r') as archivo:
        config = yaml.safe_load(archivo)

    # Llamar a la función de entrenamiento con los argumentos
    main(config, args.dataset, args.key, args.cross_val)