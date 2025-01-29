import torch
import numpy as np
import os
import argparse
import wandb

from models.dnn import FCNN, CNN
from models.vlaai import VLAAI
from models.vlaai_pytorch import VLAAI as VLAAI_pytorch
from models.eeg_conformer import Conformer, ConformerConfig

# turn a tensor to 0 mean and std of 1 with shape (C, T) and return shape (C)   
def normalize_eeg(tensor: torch.tensor):

    # unsqueeze necesario para el broadcasting (C) => (C, 1)
    mean = (torch.mean(tensor, dim=1)).unsqueeze(1)
    std = torch.std(tensor, dim=1).unsqueeze(1)

    return (tensor - mean) / std

# turn a tensor to 0 mean and std of 1 with shape (T)
def normalize_stim(tensor: torch.tensor):

    mean = torch.mean(tensor)
    std = torch.std(tensor)

    return (tensor - mean) / std

jaulab_fixed_trials = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 88, 89, 90, 91]
# Return the required trials for splitting correctly the dataset introducing the set
def get_trials(split: str, n_trials: int, cv_fold: int = None, shuffle: bool = False, fixed:bool = False, dataset:str = None):

    if cv_fold is None:
        partitions = [0.8, 0.10, 0.10] # sum to 1

        if fixed:
            n_trials = len(jaulab_fixed_trials)
            trials = jaulab_fixed_trials
        else:
            trials = np.arange(0, n_trials)

        sizes = [int(n_trials * p) for p in partitions]
        sizes[0] = n_trials - sum(sizes[1:]) # Add resting samples to the train set
        
        # Get random indices from a file containing the indices generated randomly from 0 to n_trials
        if shuffle:
            assert(dataset is not None)
            seed = 1
            if not fixed:
                path = 'rnd_indices/'+dataset+'_shuffle_idx_seed='+str(seed)
            else:
                path = 'rnd_indices/'+dataset+'_fixed_shuffle_idx_seed='+str(seed)
            indices = torch.load(path)
        else:
            indices = np.arange(0, n_trials)

        if split == 'train':
            return  np.array([trials[idx] for idx in indices[:sizes[0]]])
        elif split == 'val':
            return  np.array([trials[idx] for idx in indices[sizes[0]:n_trials-sizes[1]]])
        elif split == 'test':
            return  np.array([trials[idx] for idx in indices[n_trials-sizes[1]:n_trials]])
        elif split == 'all':
            return  np.array([trials[idx] for idx in indices])
        else:
            raise ValueError('Field split must be a train/val/test/all value')

    # 5 cross validation fold
    else: 
        trials_per_fold = n_trials // 5
        blocks = np.array([np.arange(n * trials_per_fold, (n + 1) * trials_per_fold) for n in range(5)])
        blocks = np.roll(blocks, cv_fold, axis=0)

        if split == 'train':
            return np.concatenate(blocks[:3])
        elif split == 'val':
            return blocks[3]
        elif split == 'test':
            return blocks[4]
        elif split == 'all':
            return np.concatenate(blocks)
        else:
            raise ValueError('Field split must be a train/val/test/all value')
        
# Return the required trials for splitting correctly the dataset introducing the set when population
def get_leave_one_out_trials(split: str, n_trials: int, alternate: bool = False, fixed:bool = False):

    if fixed:
        n_trials = len(jaulab_fixed_trials)
        trials = jaulab_fixed_trials
    else:
        trials = np.arange(0, n_trials)
    
    # When train mode return all trials corresponding to the n-1 subjects
    if split == 'train':
        return np.array([trials[idx] for idx in range(n_trials)])
    
    # When val or test model return half of the trials for each mode corresponding to the leave out subject during training
    elif split == 'val':
        if not alternate: # When alternate activated select even samples
            return np.array([trials[idx] for idx in range(n_trials//2)])
        else: # When alternate disabled select the first half
            return np.array([trials[idx] for idx in range(0, n_trials, 2)])
    elif split == 'test':
        if not alternate: # When alternate activated select odd samples
            return np.array([trials[idx] for idx in range(n_trials//2, n_trials)])
        else: # When alternate disabled select the second half
            return np.array([trials[idx] for idx in range(1, n_trials, 2)])
    else:
        raise ValueError('Field split must be a train/val/test value')

def verbose(mode, key, subj, dataset, model, window=None, loss_mode=None):

    if mode == 'train':
        prefix = 'Training'
    elif mode == 'finetune':
        prefix = 'Finetunning'
    elif mode == 'evaluate':
        prefix = f'Evaluating on window {window}'
    else: raise ValueError('Mode must be train/finetune/evaluate')

    if key == 'population':
        print(f'{prefix} {model} on all subjects with {dataset} data...')
    elif key == 'subj_independent':
        print(f'{prefix} {model} leaving out {subj} with {dataset} data...')
    elif key == 'subj_specific':
        print(f'{prefix} {model} on {subj} with {dataset} data...')
    else: raise ValueError('Key must be population/subj_specific/subj_independent')

    if loss_mode is not None:
        print(f'Using criterion: {loss_mode}')

# introduce subject index like 'S1' and return index like 'sub-001'
def get_SKL_subj_idx(subject):
    idx = subject[1:]
    zeros = 3 - len(idx) # number of zeros you have to add to the idx
    subj_idx = 'sub-' + ''.join(['0' for _ in range(zeros)] + [idx])
    return subj_idx

# Returns the subjects not present in the list
def get_other_subjects(subject, dataset):
    
    # Obtain the remaining subject on the population setting for saving the results
    # BUG: in the case of the dataset jaulab, subjects 13 and 16 aren't used because of the electrodes used
    # so delete them from the excluded subjects list adn differ it when defining the number of total subjects
    # jaulab_bug_subj = [13, 16]
    ds_subjects = {'fulsang': ['S'+str(n) for n in range(1, 19)], 
                #    'jaulab': ['S'+str(n) for n in range(1, 18) if n not in jaulab_bug_subj],
                   'jaulab': ['S'+str(n) for n in range(1, 18)],
                   'skl': ['S'+str(n) for n in range(1, 41)]}
    other_subjects = list(set(ds_subjects[dataset]) - set(subject))
    return other_subjects

def add_appendix(name, appendix):
    name += '_' + str(appendix)
    return name
    
# Defines a model filename for saving and loading the model with all the config features
def get_mdl_name(config):

    model = config['model']
    train_config = config['train_params']
    dataset_config = config['dataset_params']
    loss_config = config['loss_params']
    mdl_config = config['model_params']

    mdl_name = f'{model}_batch={train_config["batch_size"]}_block={dataset_config["window"]}_lr={train_config["lr"]}'
    
    if mdl_config.get('dropout'): mdl_name = add_appendix(mdl_name, 'dr=' + str(mdl_config.get('dropout')))
    
    # Add extensions to the model name depending on the params
    if train_config.get('preproc_mode'): mdl_name = add_appendix(mdl_name, train_config.get('preproc_mode'))
    if dataset_config.get('eeg_band'): mdl_name = add_appendix(mdl_name, dataset_config.get('eeg_band'))
    if loss_config.get('mode') != 'mean': mdl_name = add_appendix(mdl_name, loss_config.get('mode'))
    if loss_config.get('alpha_end'): mdl_name = add_appendix(mdl_name, 'alpha=' + str(loss_config.get('alpha_end')))
    
    if dataset_config.get('hrtf'): mdl_name = add_appendix(mdl_name, 'hrtf')
    if dataset_config.get('norm_hrtf_diff'): mdl_name = add_appendix(mdl_name, 'norm_diff')
    if dataset_config.get('fixed'): mdl_name = add_appendix(mdl_name, 'fixed')
    if dataset_config.get('rnd_trials'): mdl_name = add_appendix(mdl_name, 'rnd_trials')
    if train_config.get('shuffle'): mdl_name = add_appendix(mdl_name, 'shuffle')

    return mdl_name

# Compute the interaural level difference ILD (dB)
def compute_ild(left_channel, right_channel):
    # Calculate RMS for each channel
    rms_left = np.sqrt(np.mean(left_channel**2))
    rms_right = np.sqrt(np.mean(right_channel**2))
    # Calculate ILD in dB
    ild = 10 * np.log10(rms_left / rms_right)
    return ild

def load_model(config_run, dataset, wandb_upload):

    if config_run['model'] == 'FCNN':
        config_run['model_params']['n_chan'] = get_channels(dataset)
        mdl = FCNN(**config_run['model_params'])

    elif config_run['model'] == 'CNN':
        if wandb_upload:
            # Sweep params implemented
            mdl_config = config_run['model_params']
            mdl_config['dropout'] = getattr(wandb.config, 'dropout', mdl_config.get('dropout'))
            mdl_config['input_samples'] = getattr(wandb.config, 'input_samples', mdl_config.get('input_samples'))
            mdl_config['F1'] = getattr(wandb.config, 'F1', mdl_config.get('F1'))
            mdl_config['D'] = getattr(wandb.config, 'D', mdl_config.get('D'))
            mdl_config['AP1'] = getattr(wandb.config, 'AP1', mdl_config.get('AP1'))
            mdl_config['AP2'] = getattr(wandb.config, 'AP2', mdl_config.get('AP2'))
            config_run['model_params']['input_channels'] = get_channels(dataset)
        mdl = CNN(**config_run['model_params'])

    elif config_run['model'] == 'VLAAI':
        config_run['model_params']['input_channels'] = get_channels(dataset)
        mdl = VLAAI(**config_run['model_params'])

    elif config_run['model'] == 'VLAAI_pytorch':
        config_run['model_params']['input_channels'] = get_channels(dataset)
        mdl = VLAAI_pytorch(**config_run['model_params'])

    elif config_run['model'] == 'Conformer':
        config_run['model_params']['eeg_channels'] = get_channels(dataset)
        config_run['model_params']['kernel_chan'] = get_channels(dataset)
        mdl_config = ConformerConfig(**config_run['model_params'])
        mdl = Conformer(mdl_config)

    else:
        raise ValueError('Introduce a valid model')
    
    return mdl
    
# Returns the filename related to the subject solving the problem of S1
def get_filename(mdl_folder_path, cv_fold=None):
    
    list_dir = os.listdir(mdl_folder_path)
    
    if cv_fold is None:
        for file in list_dir:
            if cv_prefix in file:
                return file
        raise ValueError(f'The file on the folder {mdl_folder_path} was not found')
    
    else:
        cv_prefix = f'cvFold={cv_fold}'
        for file in list_dir:
            if cv_prefix in file:
                return file
        raise ValueError(f'The file on the folder {mdl_folder_path} with cv_fold {cv_fold} was not found')

# Return used datapaths
def get_data_path(global_data_path:str, dataset:str, preproc_mode=None):
    
    paths = {'hugo_path': "/Hugo_2022/hugo_preproc_data",
            'fulsang_path': '/Fulsang_2017/DATA_preproc',
            'fulsang_upsample_path': '/Fulsang_2017/DATA_preproc_256Hz',
            'fulsang_thorton_path': '/Fulsang_2017/DATA_preproc_Thorton',
            'fulsang_thortonF_path': '/Fulsang_2017/DATA_fulsPrep_thortF',
            'fulsang_thortonFN_path': '/Fulsang_2017/DATA_fulsPrep_thortFN',
            'fulsang_bandAnalysis_path': '/Fulsang_2017/DATA_fulsPrep_bandAnalysis',
            'jaulab_path': '/Jaulab_2024/PreprocData_ICA',
            'jaulab_filt_path': '/Jaulab_2024/DATA_filtered',
            'jaulab_fix_path': '/Jaulab_2024/fixed_trials.npy',
            'skl_path': '/SKL_2023/split_data',
            'skl_filt_path': None, 
    }

    folder_path = paths[dataset+'_'+preproc_mode+'_path'] if preproc_mode is not None else paths[dataset+'_path']
    return global_data_path + folder_path

def get_channels(dataset:str):
    channels = {
        'skl': 64,
        'fulsang': 64,
        'jaulab': 61
    }
    return channels[dataset]

def get_subjects(dataset:str):
    subjects = {
        'fulsang': ['S'+str(n) for n in range(1, 19)],
        'skl': ['S'+str(n) for n in range(1, 85)],
        # 'jaulab' : ['S'+str(n) for n in range(1, 18) if n not in jaulab_excl_subj]
        'jaulab' : ['S'+str(n) for n in range(1, 18)]
    }
    return subjects[dataset]

def get_trials_len(dataset:str):
    trial_lenghts = {
        'skl': 3200,
        'fulsang': 3200,
        'jaulab': 1696
    }
    return trial_lenghts[dataset]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def set_seeds(seed: int=42):
    """ Set random seeds for torch operations

    Args:
        seed (int, optional): Random seed to set. Defaults 42.

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def multiple_loss_opt(loss_mode:str):
    return loss_mode in ['ressamble', 'corr_ild_mae', 'corr_ild_mse', 'corr_ild_mse_penalty', 'corr_ild_mse_penalty_w', 'corr_diff_mse', 'corr_diff_mae']