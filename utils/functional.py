import torch
import numpy as np
import os
import argparse

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
def get_trials(split: str, n_trials: int, shuffle: bool = False, fixed:bool = False, dataset:str = None):

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
        return np.array([trials[idx] for idx in indices[:sizes[0]]])
    elif split == 'val':
        return np.array([trials[idx] for idx in indices[sizes[0]:n_trials-sizes[1]]])
    elif split == 'test':
        return np.array([trials[idx] for idx in indices[n_trials-sizes[1]:n_trials]])
    elif split == 'all':
        return np.array([trials[idx] for idx in indices])
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

# Calculates the pearson correlation between two tensors
def correlation(x: torch.tensor, y: torch.tensor, eps=1e-8, batch_dim=True):

    """Correlation function that returns calculation the Person's coef
    
    Params
    ----------

    x, y: tensor
        two vectors for calculating the correlation, must have the same dims

    eps: 1e-8 by default applied on the formula
    
    batch_dim: bool
        select wether you're introducing vectors to compute the corr on batch_dim (default: True)
        or you calculate multiple correlations for the batch_dim and then compute the mean

    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    if batch_dim:
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    else:
        corr = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + eps)
        corr = torch.mean(corr)

        # IS THE SAME AS GETTING IT ON BATCH DIM AND THEN COMPUTE THE MEAN
        # corr_list = torch.zeros((x.shape[0]))
        # for n, (x_b, y_b) in enumerate(zip(x, y)):
        #     vx_b = x_b - torch.mean(x_b)
        #     vy_b = y_b - torch.mean(y_b)

        #     corr_b = torch.sum(vx_b * vy_b) / (torch.sqrt(torch.sum(vx_b ** 2)) * torch.sqrt(torch.sum(vy_b ** 2)) + eps)
        #     corr_list[n] = corr_b
        # corr = torch.mean(corr_list)

    return corr
    
# Returns the filename related to the subject solving the problem of S1
def get_filename(mdl_folder_path, subject):
    list_dir = os.listdir(mdl_folder_path)
    filename = ''
    for file in list_dir:
        if subject in file:
            if subject == 'S1':
                idx = file.index(subject)
                if file[idx+2] == '_': # si el siguiente caracter al S1 es un barra baja añade al diccionario
                    filename = file
            else:
                filename = file
    return filename

# Return datapath
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

