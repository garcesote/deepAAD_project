import torch
import numpy as np
import os
import argparse
import wandb
import random

from models.dnn import FCNN, CNN
from models.vlaai import VLAAI
from models.vlaai_pytorch import VLAAI as VLAAI_pytorch
from models.eeg_conformer import Conformer, ConformerConfig
from models.add_net import AAD_Net, AAD_Net_Config
from models.triplet_net import Triplet_Net, Triplet_Net_Config

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
            if dataset != 'kuleuven':
                trials = np.arange(0, n_trials)
            else: 
                trials = np.arange(0, 20)

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

        if dataset != 'kuleuven':
            if split == 'train': idx_range = range(sizes[0])
            elif split == 'val': idx_range = range(sizes[0],n_trials-sizes[1])
            elif split == 'test': idx_range = range(n_trials-sizes[1],n_trials)
            elif split == 'all': idx_range = range(n_trials)
            else: raise ValueError('Field split must be a train/val/test/all value')

            return np.array([trials[idx] for idx in indices[idx_range]])
        
        else: 
            # KULeuven that needs a mapping for tha last indices to the corresponding trials (3 trials each last idx)
            idx_to_trials = [n for n in range(8)]
            for idx in range(8, 12):
                idx_to_trials.extend([idx for n in range(3)])
            idx_to_trials = np.array(idx_to_trials)  # [1, 2, ..., 8, 8, 8, ..., 11, 11, 11]
            if split == 'train': idx_range = range(sizes[0])
            elif split == 'val': idx_range = range(sizes[0],n_trials-sizes[1])
            elif split == 'test': idx_range = range(n_trials-sizes[1],n_trials)
            elif split == 'all': idx_range = range(n_trials)
            else: raise ValueError('Field split must be a train/val/test/all value')
            
            indices_list = np.concatenate([
                np.where(idx_to_trials == idx)[0] # [0] because where retuns a tuple
                for idx in indices[idx_range]
            ])
            return trials[indices_list]

            
    # 5-fold cross validation (fulsang) or 4-fold cross validation (kuleuven/jaulab) scheme
    else: 

        n_folds = 5 if dataset == 'fulsang' else 4
        trials_per_fold = n_trials // n_folds

        if dataset != 'kuleuven':
            blocks = [np.arange(n * trials_per_fold, (n + 1) * trials_per_fold) for n in range(n_folds)]
            
        
        else: # trial mapping
            idx_to_trials = [n for n in range(8)]
            for idx in range(8, 12):
                idx_to_trials.extend([idx for n in range(3)])
            idx_to_trials = np.array(idx_to_trials)  # [1, 2, ..., 8, 8, 8, ..., 11, 11, 11]
            blocks = []
            for n in range(n_folds):
                indices = np.arange(n * trials_per_fold, (n + 1) * trials_per_fold)
                trials = np.concatenate([
                    np.where(idx_to_trials == idx)[0]
                    for idx in indices
                ])
                blocks.append(trials.tolist())

        # Roll the blocks depending on the fold
        blocks = blocks[-cv_fold:] + blocks[:-cv_fold]
        
        if split == 'train': return np.concatenate(blocks[:n_folds-2])
        elif split == 'val': return blocks[n_folds-2]
        elif split == 'test': return blocks[n_folds-1]
        elif split == 'all': return np.concatenate(blocks)
        else: raise ValueError('Field split must be a train/val/test/all value')
        
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

def verbose(mode, key, subj, dataset, model, window=None, loss_mode=None, cv_fold=None):

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
        print(f'Using criterion: {loss_mode} | CV fold: {cv_fold}')

# introduce subject index like 'S1' and return index like 'sub-001'
def get_SKL_subj_idx(subject):
    idx = subject[1:]
    zeros = 3 - len(idx) # number of zeros you have to add to the idx
    subj_idx = 'sub-' + ''.join(['0' for _ in range(zeros)] + [idx])
    return subj_idx

def add_appendix(name, appendix):
    name += '_' + str(appendix)
    return name
    
# Defines a model filename for saving and loading the model with all the config features
def get_mdl_name(config, linear = False):
    
    train_config = config.get('train_params')
    dataset_config = config.get('dataset_params')
    mdl_config = config.get('model_params')

    if not linear:
        model = config['model']
        loss_config = config['loss_params']
        mdl_name = f'{model}_batch={train_config["batch_size"]}_block={dataset_config["window"]}_lr={train_config["lr"]}'
        if loss_config.get('mode') != 'mean': mdl_name = add_appendix(mdl_name, loss_config.get('mode'))
        if loss_config.get('alpha_end'): mdl_name = add_appendix(mdl_name, 'alpha=' + str(loss_config.get('alpha_end')))
    else:
        linear_model = config['linear_model']
        if linear_model == 'CCA':
            mdl_name = f"{linear_model}_enc={mdl_config['encoder_len']}_dec={mdl_config['decoder_len']}_comp={mdl_config['n_components']}"
        else:
            mdl_name = f"{linear_model}_start={mdl_config['start_lag']}_end={mdl_config['end_lag']}"

    if mdl_config.get('dropout'): mdl_name = add_appendix(mdl_name, 'dr=' + str(mdl_config.get('dropout')))
    if mdl_config.get('max_iter'): mdl_name = add_appendix(mdl_name, 'max_iter=' + str(mdl_config['max_iter']))
    if mdl_config.get('tol'): mdl_name = add_appendix(mdl_name, 'tol=' + str(mdl_config['tol']))
    if mdl_config.get('per_trial'): mdl_name = add_appendix(mdl_name, 'per_trial')
    if mdl_config.get('post_stim'): mdl_name = add_appendix(mdl_name, 'post_stim=' + str(mdl_config['post_stim']))
    
    # Add extensions to the model name depending on the params
    if train_config.get('preproc_mode'): mdl_name = add_appendix(mdl_name, train_config.get('preproc_mode'))
    if dataset_config.get('eeg_band'): mdl_name = add_appendix(mdl_name, dataset_config.get('eeg_band'))
    if dataset_config.get('hrtf'): mdl_name = add_appendix(mdl_name, 'hrtf')
    if dataset_config.get('norm_hrtf_diff'): mdl_name = add_appendix(mdl_name, 'norm_diff')
    if dataset_config.get('fixed'): mdl_name = add_appendix(mdl_name, 'fixed')
    if dataset_config.get('rnd_trials'): mdl_name = add_appendix(mdl_name, 'rnd_trials')
    if train_config.get('shuffle'): mdl_name = add_appendix(mdl_name, 'shuffle')
    if train_config.get('path_dataset'): mdl_name = add_appendix(mdl_name, 'path_dataset')

    return mdl_name

# Compute the interaural level difference ILD (dB)
def compute_ild(left_channel, right_channel):
    # Calculate RMS for each channel
    rms_left = np.sqrt(np.mean(left_channel**2))
    rms_right = np.sqrt(np.mean(right_channel**2))
    # Calculate ILD in dB
    ild = 10 * np.log10(rms_left / rms_right)
    return ild

def load_model(config_run, dataset, wandb_upload, sweep=True):

    if config_run['model'] == 'FCNN':
        config_run['model_params']['n_chan'] = get_channels(dataset)
        mdl = FCNN(**config_run['model_params'])

    elif config_run['model'] == 'CNN':
        
        mdl_config = config_run['model_params'].copy()

        if wandb_upload and sweep:
            # Sweep params implemented
            mdl_config['dropout'] = getattr(wandb.config, 'dropout', mdl_config.get('dropout'))
            mdl_config['input_samples'] = getattr(wandb.config, 'input_samples', mdl_config.get('input_samples'))
            mdl_config['F1'] = getattr(wandb.config, 'F1', mdl_config.get('F1'))
            mdl_config['D'] = getattr(wandb.config, 'D', mdl_config.get('D'))
            mdl_config['AP1'] = getattr(wandb.config, 'AP1', mdl_config.get('AP1'))
            mdl_config['AP2'] = getattr(wandb.config, 'AP2', mdl_config.get('AP2'))
        
        mdl_config['input_channels'] = get_channels(dataset)
        mdl = CNN(**mdl_config)

    elif config_run['model'] == 'VLAAI':
        
        mdl_config = config_run['model_params'].copy()

        mdl_config['input_channels'] = get_channels(dataset)
        
        if wandb_upload and sweep:
            # Sweep params implemented
            mdl_config['n_blocks'] = getattr(wandb.config, 'n_blocks', mdl_config.get('n_blocks'))
            mdl_config['use_skip'] = getattr(wandb.config, 'use_skip', mdl_config.get('use_skip'))
            mdl_config['dropout'] = getattr(wandb.config, 'dropout', mdl_config.get('dropout'))
            mdl_config['output_context_kernel'] = getattr(wandb.config, 'output_context_kernel', mdl_config.get('output_context_kernel'))
            mdl_config['stack_model_kernel'] = getattr(wandb.config, 'stack_model_kernel', mdl_config.get('stack_model_kernel'))
            mdl_config['stack_model_filters'] = getattr(wandb.config, 'stack_model_filters', mdl_config.get('stack_model_filters'))
        
        mdl = VLAAI(**mdl_config)

    elif config_run['model'] == 'VLAAI_pytorch':
        config_run['model_params']['input_channels'] = get_channels(dataset)
        mdl = VLAAI_pytorch(**config_run['model_params'])

    elif config_run['model'] == 'Conformer':
        
        mdl_config = config_run['model_params'].copy()
        mdl_config['eeg_channels'] = get_channels(dataset)
        mdl_config['kernel_chan'] = get_channels(dataset)

        mdl_config = ConformerConfig(**mdl_config)
        
        if wandb_upload and sweep:
            # Sweep params implemented
            mdl_config.dropout = getattr(wandb.config, 'dropout', mdl_config.dropout)
            mdl_config.dropout_clsf = getattr(wandb.config, 'dropout_clsf', mdl_config.dropout_clsf)
            mdl_config.enc_layers = getattr(wandb.config, 'enc_layers', mdl_config.enc_layers)
            mdl_config.n_embd = getattr(wandb.config, 'n_embd', mdl_config.n_embd)
            mdl_config.pool = getattr(wandb.config, 'pool', mdl_config.pool)
            mdl_config.pool_hop = getattr(wandb.config, 'pool_hop', mdl_config.pool_hop)
            mdl_config.hidden_size = getattr(wandb.config, 'hidden_size', mdl_config.hidden_size)
        
        mdl = Conformer(mdl_config)

    elif config_run['model'] == 'AAD_Net':

        mdl_config = config_run['model_params'].copy()
        mdl_config['eeg_chan'] = get_channels(dataset)

        mdl_config = AAD_Net_Config(**mdl_config)

        if wandb_upload and sweep:
            # Sweep params implemented
            mdl_config.eeg_transform = getattr(wandb.config, 'eeg_transform', mdl_config.eeg_transform)
            mdl_config.eeg_out_feat = getattr(wandb.config, 'eeg_out_feat', mdl_config.eeg_out_feat)
            mdl_config.eeg_feat_kernel = getattr(wandb.config, 'eeg_feat_kernel', mdl_config.eeg_feat_kernel)
            
            mdl_config.stim_transform = getattr(wandb.config, 'stim_transform', mdl_config.stim_transform)
            mdl_config.stim_out_feat = getattr(wandb.config, 'stim_out_feat', mdl_config.stim_out_feat)
            mdl_config.stim_feat_kernel = getattr(wandb.config, 'stim_feat_kernel', mdl_config.stim_feat_kernel)
            
            mdl_config.eeg_pool = getattr(wandb.config, 'eeg_pool', mdl_config.eeg_pool)
            mdl_config.stim_pool = getattr(wandb.config, 'stim_pool', mdl_config.stim_pool)

        mdl = AAD_Net(mdl_config)

    elif config_run['model'] == 'Triplet_Net':

        mdl_config = config_run['model_params'].copy()
        mdl_config['eeg_chan'] = get_channels(dataset)

        mdl_config = Triplet_Net_Config(**mdl_config)

        if wandb_upload and sweep:

            # Sweep params implemented
            mdl_config.n_emb = getattr(wandb.config, 'n_emb', mdl_config.n_emb)
            mdl_config.margin = getattr(wandb.config, 'margin', mdl_config.margin)
            mdl_config.eeg_ks = getattr(wandb.config, 'eeg_ks', mdl_config.eeg_ks)
            mdl_config.stim_conv_ks = getattr(wandb.config, 'stim_conv_ks', mdl_config.stim_conv_ks)
            mdl_config.skip = getattr(wandb.config, 'skip', mdl_config.skip)
            
            # Si stim_architecture existe en wandb.config, extrae los valores específicos
            if hasattr(wandb.config, 'stim_architecture'):
                mdl_config.stim_features = wandb.config.stim_architecture['features']
                mdl_config.stim_pool = wandb.config.stim_architecture['pool']            

        mdl = Triplet_Net(mdl_config)

    else:
        raise ValueError('Introduce a valid model')
    
    return mdl
    
# Returns the filename
def get_filename(mdl_folder_path, cv_fold=None):
    
    list_dir = os.listdir(mdl_folder_path)
    
    if cv_fold is None:
        for file in list_dir:
            if 'cvFold' not in file:
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
            'jaulab_path': '/Jaulab_2024/PreprocData_COCOHA_26',
            'jaulab_filt_path': '/Jaulab_2024/DATA_filtered',
            'jaulab_fix_path': '/Jaulab_2024/fixed_trials.npy',
            'kuleuven_path': '/KULeuven_2016/preprocessed_data_py',
            'skl_path': '/SKL_2023/split_data',
            'skl_filt_path': None, 
    }

    folder_path = paths[dataset+'_'+preproc_mode+'_path'] if preproc_mode is not None else paths[dataset+'_path']
    return global_data_path + folder_path

def get_channels(dataset:str):
    channels = {
        'skl': 64,
        'fulsang': 64,
        'kuleuven': 64,
        'jaulab': 61
    }
    return channels[dataset]

def get_subjects(dataset:str):
    jaulab_bug_subj = [13, 16]
    subjects = {
        'fulsang': ['S'+str(n) for n in range(1, 19)],
        'skl': ['S'+str(n) for n in range(1, 86)],
        # 'jaulab' : ['S'+str(n) for n in range(1, 18) if n not in jaulab_bug_subj],
        'jaulab' : ['S'+str(n) for n in range(1, 25)],
        'kuleuven' : ['S'+str(n) for n in range(16, 17)],
    }
    return subjects[dataset]

# Returns the subjects not present in the list
def get_other_subjects(subject, dataset):
    # Obtain the remaining subject on the population setting for saving the results
    # BUG: in the case of the dataset jaulab, subjects 13 and 16 aren't used because of the electrodes used
    # so delete them from the excluded subjects list adn differ it when defining the number of total subjects
    jaulab_bug_subj = [13, 16]
    ds_subjects = {'fulsang': ['S'+str(n) for n in range(1, 19)], 
                #    'jaulab': ['S'+str(n) for n in range(1, 18) if n not in jaulab_bug_subj],
                   'jaulab': ['S'+str(n) for n in range(1, 25)],
                   'skl': ['S'+str(n) for n in range(1, 41)],
                   'kuleuven' : ['S'+str(n) for n in range(1, 17)],
    }
    other_subjects = list(set(ds_subjects[dataset]) - set(subject))
    return other_subjects

def get_trials_len(dataset:str):
    trial_lenghts = {
        'skl': 3200,
        'fulsang': 3200,
        'kuleuven': 23040,
        'jaulab': 1664,
    }
    return trial_lenghts[dataset]

def get_n_trials(dataset:str):
    trial_lenghts = {
        'skl': 4,
        'fulsang': 60,
        'kuleuven': 12, # 12 trials considered to map the 20 real trials
        'jaulab': 96,
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
    random.seed(seed)

def multiple_loss_opt(loss_mode:str):
    return loss_mode in ['ressamble', 'corr_ild_mae', 'corr_ild_mse', 'corr_ild_mse_penalty', 'corr_ild_mse_penalty_w', 'corr_diff_mse', 'corr_diff_mae']