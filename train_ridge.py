import torch
from models.ridge import Ridge
import numpy as np
import os
import pickle
from utils.functional import get_data_path, get_trials_len, get_subjects, str2bool
from utils.datasets import CustomDataset
import argparse

def main(
        dataset: str,
        key: str,
        fixed: bool,
        rnd_trials: bool,
        preproc_mode: str = None,
        data_type: str = 'mat'
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
            print(f'Training Ridge on all subjects with {dataset} data...')
        elif key == 'subj_independent':
            print(f'Training Ridge leaving out {subj} with {dataset} data...')
        else:
            print(f'Training Ridge on {subj} with {dataset} data...')

        # LOAD THE DATA
        train_set = CustomDataset(dataset, data_path, 'train', subj, window=trial_len, hop=trial_len, data_type=data_type, 
                                    leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials)
        val_set = CustomDataset(dataset, data_path, 'val',  subj, window=trial_len, hop=trial_len, data_type=data_type, 
                                leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials)

        alphas = np.logspace(-7,7, 15)
        trial_len = get_trials_len(dataset) # in order to estumate correctly the lag matrix

        mdl = Ridge(start_lag=0, end_lag=30, alpha=alphas, trial_len = trial_len, original=False)

        train_eeg, train_stim = train_set.eeg, train_set.stima 
        val_eeg, val_stim = val_set.eeg, val_set.stima

        # TRAIN MODEL
        mdl.fit(train_eeg.T, train_stim[:, np.newaxis])
        
        # VALIDATE AND SELECT BEST ALPHA
        scores = mdl.model_selection(val_eeg.T, val_stim[:, np.newaxis])
        best_alpha = mdl.best_alpha_idx

        print(f'Ridge trained for {dataset} data with a score of {scores[best_alpha]} with alpha = {best_alpha}')

        # SAVE THE MODEL
        model_name = 'Ridge'
        # Add extensions to the model name depending on the params
        if preproc_mode is not None:
            model_name = model_name + '_' + preproc_mode
        if rnd_trials:
            model_name = model_name + '_rnd'

        dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

        mdl_prefix = key if key == 'population' else subj
        mdl_folder = os.path.join(mdl_save_path, dataset_filename + '_data', model_name)
        if not os.path.exists(mdl_folder):
            os.makedirs(mdl_folder)
        save_path = os.path.join(mdl_folder, f'{mdl_prefix}_alpha={best_alpha}_acc={scores[best_alpha]:.4f}')
        pickle.dump(mdl, open(save_path, "wb"))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training Ridge script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--fixed", type=str2bool, default='False', help="Static Jaulab trials")
    parser.add_argument("--rnd_trials", type=str2bool, default='False', help="Random trial selection")
    parser.add_argument("--preproc_mode", type=str, default=None, help="Random trial selection")
    parser.add_argument("--data_type", type=str, default='mat', help="Random trial selection")

    args = parser.parse_args()

    # Llamar a la función de entrenamiento con los argumentos
    main(
        args.dataset,
        args.key,
        args.fixed,
        args.rnd_trials,
        args.preproc_mode,
        args.data_type
    )