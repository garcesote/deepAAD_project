import torch
from models.ridge import Ridge
import numpy as np
import os
import pickle
from utils.functional import check_jaulab_chan
from utils.datasets import CustomDataset
import argparse

def main(
        dataset: str,
        key: str,
        filt: bool,
        fixed: bool,
        rnd_trials: bool,
    ):

    # Saving path parameters
    global_path = 'C:/Users/jaulab/Desktop/AAD/IA_poster'
    mdl_save_path = global_path + '/Results/'+key+'/models'

    # Select subjects you want your network to be trained on
    subjects = {
        'fulsang_subj': ['S'+str(n) for n in range(1, 19)],
        'skl_subj': ['S'+str(n) for n in range(1, 85)],
        # 'jaulab_subj' : ['S'+str(n) for n in range(1, 18) if n not in jaulab_excl_subj]
        'jaulab_subj' : ['S'+str(n) for n in range(1, 18)]
    }

    # Data path parameters
    global_path = 'C:/Users/jaulab/Desktop/AAD/Data'
    # global_path = 'C:/Users/garcia.127407/Desktop/DNN_AAD/Data'

    paths = {'hugo_path': global_path + "/Hugo_2022/hugo_preproc_data",
            'fulsang_path': global_path + '/Fulsang_2017/DATA_preproc',
            'jaulab_path': global_path + '/Jaulab_2024/PreprocData_ICA',
            'fulsang_filt_path': global_path + '/Fulsang_2017/DATA_filtered',
            'jaulab_filt_path': global_path + '/Jaulab_2024/DATA_filtered',
            'jaulab_fix_path': global_path + '/Jaulab_2024/fixed_trials.npy',
            'skl_path': global_path + '/SKL_2023/split_data',
            'skl_filt_path': None,
    }

    trial_lenghts = {
        'skl': 3200,
        'fulsang': 3200,
        'jaulab': 1696
    }

    #------------------------------------------------------------------------------------------

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
    population = True if key == 'subj_independent' else False # Attention! This parameter change the whole sim. (read doc)
    filt = filt
    fixed = fixed
    rnd_trials = rnd_trials
    trial_len = trial_lenghts[dataset]

    
    #------------------------------------------------------------------------------------------

    if key == 'population':
        selected_subj = [subjects[dataset+'_subj']]
    else:
        selected_subj = subjects[dataset+'_subj']

    for subj in selected_subj:
        
        if key == 'population':
            print(f'Training Ridge on all subjects with {dataset} data...')
        elif key == 'subj_independent':
            print(f'Training Ridge leaving out {subj} with {dataset} data...')
        else:
            print(f'Training Ridge on {subj} with {dataset} data...')

        # LOAD THE DATA
        train_set = CustomDataset(dataset, paths[dataset+'_path'], 'train', subj, window=trial_len, hop=trial_len, filt=filt, filt_path=paths[dataset+'_filt_path'], 
                                    population=population, fixed=fixed, rnd_trials = rnd_trials)
        val_set = CustomDataset(dataset, paths[dataset+'_path'], 'val',  subj, window=trial_len, hop=trial_len, filt=filt, filt_path=paths[dataset+'_filt_path'], 
                                population=population, fixed=fixed, rnd_trials = rnd_trials)

        alphas = np.logspace(-7,7, 15)
        trial_len = trial_lenghts[dataset]

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
        if filt:
            model_name = model_name + '_filt'
        if rnd_trials:
            model_name = model_name + '_rnd'

        mdl_prefix = key if key == 'population' else subj
        mdl_folder = os.path.join(mdl_save_path, dataset + '_data', model_name)
        if not os.path.exists(mdl_folder):
            os.makedirs(mdl_folder)
        save_path = os.path.join(mdl_folder, f'{mdl_prefix}_alpha={best_alpha}_acc={scores[best_alpha]:.4f}')
        pickle.dump(mdl, open(save_path, "wb"))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--filt", type=str, default='False', help="EEG filtered")
    parser.add_argument("--fixed", type=str, default='False', help="Static Jaulab trials")
    parser.add_argument("--rnd_trials", type=str, default='False', help="Random trial selection")
    
    # Parsear los argumentos
    args = parser.parse_args()
    print(args)
    filt = False if args.filt == 'False' else True
    fixed = False if args.fixed == 'False' else True
    rnd_trials = False if args.rnd_trials == 'False' else True
    
    # Llamar a la función de entrenamiento con los argumentos
    main(
        args.dataset,
        args.key,
        filt,
        fixed,
        rnd_trials
    )