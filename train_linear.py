import torch
from models.ridge import Ridge
from models.cca import CCA_AAD
import numpy as np
import os
import pickle
from utils.functional import get_data_path, get_trials_len, get_subjects, str2bool
from utils.datasets import CustomDataset
import argparse

def main(
        dataset: str,
        key: str,
        linear_model: str,
        fixed: bool,
        rnd_trials: bool,
        preproc_mode: str = None,
        data_type: str = 'mat',
        eeg_band: str = None
    ):

    # Saving path parameters
    global_path = 'C:/Users/jaulab/Desktop/deepAAD_project'
    global_data_path = 'C:/Users/jaulab/Desktop/AAD/Data'
    # global_path: 'C:/Users/garcia.127407/Desktop/DNN_AAD/deepAAD_project'
    # global_data_path: 'D:\igarcia\AAD_Data'
    mdl_save_path = global_path + '/results/'+key+'/models'
    data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)

    """

    Training parameters
    ------------------

    dataset: string
        select a dataset between 'skl', 'fulsang' or 'jaulab' to train the network on

    population: bool
        select if the model would be trained on the leave-one-out subject paradigm (True) 
        or on the specified subject (False) (subject specific/ subject independent)

    filt: bool
        select wether you want to select the filtered eeg from fulsang or jaulab
    
    fixed: bool
        in the case the dataset is "jaulab" select only the trials in which the stimulus is 
        fixed during the experiment. 

    rnd_trials: bool
        select if you want your trials to be selected randomly or assing them in order.
        In subject-specific mode trials are shuffled while in population mode, trials for eval.
        the excluded subject (val and test) are selected one-by-one.

    """

    dataset = dataset
    leave_one_out = True if key == 'subj_independent' else False # Attention! This parameter change the whole sim. (read doc)
    fixed = fixed
    rnd_trials = rnd_trials
    trial_len = get_trials_len(dataset)

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
                                    leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, eeg_band=eeg_band)
        val_set = CustomDataset(dataset, data_path, 'val',  subj, window=trial_len, hop=trial_len, data_type=data_type, 
                                leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, eeg_band=eeg_band)

        if linear_model == 'Ridge':

            alphas = np.logspace(-7,7, 15)
            mdl = Ridge(start_lag=-25, end_lag=0, alpha=alphas, trial_len = trial_len, original=False)

            # TRAIN MODEL
            mdl.fit(train_set.eeg.T, train_set.stima[:, np.newaxis])
            
            # VALIDATE AND SELECT BEST ALPHA
            scores = mdl.model_selection(val_set.eeg.T, val_set.stima[:, np.newaxis])
            best_alpha = mdl.best_alpha_idx

            mdl_filename = f'{mdl_prefix}_alpha={best_alpha}_acc={scores[best_alpha]:.4f}'

            print(f'Ridge trained for {dataset} data with a score of {scores[best_alpha]} with alpha = {best_alpha}')

        elif linear_model == 'CCA':
            
            encoder_len = 80 # 1.25s pre-stim lags
            decoder_len = 16 # .25s post-stim lags
            n_components = 2

            mdl = CCA_AAD(encoder_len=encoder_len, decoder_len=decoder_len, trial_len= trial_len, n_components=n_components)
            
            # TRAIN THE CCA PROJECTION MODEL
            mdl.fit_CCA(eeg=train_set.eeg, stim=train_set.stima)

            # TRAIN THE LDA CLASSIFIER MODEL
            mdl.fit_LDA(eeg=train_set.eeg, stima=train_set.stima, stimb=train_set.stimb, batch_size=trial_len)
            
            # VALIDATE THE MODEL
            scores_a = mdl.score_in_batches(val_set.eeg, val_set.stima, batch_size=trial_len)
            scores_b = mdl.score_in_batches(val_set.eeg, val_set.stimb, batch_size=trial_len)
            accuracy = mdl.classify_in_batches(val_set.eeg, val_set.stima, val_set.stimb, batch_size=trial_len)

            mdl_filename = f'{mdl_prefix}_components={n_components}_acc={accuracy}'

            print(f'CCA stats on {dataset} dataset: score_a {np.mean(scores_a)} | score_b {np.mean(scores_b)} | val_accuracy {accuracy}%')

        else: raise ValueError('Introduce a valid lineal model between Ridge or CCA')

        # SAVE THE MODEL
        model_name = linear_model
        # Add extensions to the model name depending on the params
        if preproc_mode is not None:
            model_name = model_name + '_' + preproc_mode
        if eeg_band is not None:
            model_name = model_name + '_' + eeg_band
        if rnd_trials:
            model_name = model_name + '_rnd'

        dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset
        
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
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_specific', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--linear_model", type=str, default='CCA', help="Select the linear model between Ridge or CCA")
    parser.add_argument("--fixed", type=str2bool, default='False', help="Static Jaulab trials")
    parser.add_argument("--rnd_trials", type=str2bool, default='False', help="Random trial selection")
    parser.add_argument("--preproc_mode", type=str, default=None, help="Select preprocessing mode")
    parser.add_argument("--data_type", type=str, default='mat', help="Data type between mat or npy")
    parser.add_argument("--eeg_band", type=str, default=None, help="Select the freq band (delta, theta, alpha, beta)")

    args = parser.parse_args()

    # Llamar a la función de entrenamiento con los argumentos
    main(
        args.dataset,
        args.key,
        args.linear_model,
        args.fixed,
        args.rnd_trials,
        args.preproc_mode,
        args.data_type,
        args.eeg_band
    )