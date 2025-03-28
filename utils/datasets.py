import os
import scipy
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.functional import get_other_subjects, get_trials, normalize_eeg, get_SKL_subj_idx, get_n_trials, get_trials_len, get_channels
import gc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random

class CustomDataset(Dataset):

    """ CustomDataset 
    
    Parameters
    ------------
    
    dataset:str
        intrduce a valid dataset between skl, fulsang, jaulab

    data_path:str
        path for gather the data of the dataset

    split:str
        select the split of the data between 'train', 'test', 'val'

    subjects: list, str
        select the subject / subjects you want your for your data

    window: int
        length of the window used on get_item method
    
    norm_stim: bool
        use normalized stim or not (only on fulsang or jaulab)

    leave_one_out: bool
        select if you want population mode or not
        This parameter change the whole dataset!!:
            - when True: the specified subject or subjects introduced correspond with the leaved out for training data, 
            also different trials in split (100%(rest),50%(subj),50%(subj))
            - when False: the specified subject/subjects used for all splits and the network gets trained on the specified splits (80%,10%,10%)

    fixed: bool
        in the case the dataset is "jaulab" select only the trials in which the stimulus is 
        fixed during the experiment. 

    rnd_trials: bool
        select if you want your trials to be selected randomly or assing them in order.
        In subject-specific mode trials are shuffled while in population mode, trials for eval.
        the excluded subject (val and test) are selected one-by-one.  

    window_pred: bool
        select if you want your dataset to return a stima window or unit values of stim in case
        the network returns a single value

    hrtf: bool
        when true raturn the hrtf audio envelopes with the channels on the left and right

    __len__()
    -------
    Returns the numeber samples of the whole dataset minus the length of the window for not getting
    out of range

    __getitem__()
    -------
    Returns 'window' samples of eeg separated hop samples by the next one and also the first sample
    of the attended stim.
    if acc is True then it also returns the unattended stimulus first sample
    """

    def __init__(self, dataset, data_path, split, subjects, cv_fold, window, hop, 
                 norm_stim=False, data_type = 'mat', leave_one_out = False,
                 fixed=False, rnd_trials=False, window_pred=True, hrtf=False,
                 norm_hrtf_diff=False, eeg_band= None, spatial_locus=False, stim_input = False,
                 seed=42):
        
        self.data_path = data_path
        self.dataset = dataset
        self.split = split
        self.n_subjects = len(subjects)
        self.hop = hop
        self.data_type = data_type
        self.fixed = fixed
        self.rnd_trials = rnd_trials
        self.norm_stim = norm_stim
        self.leave_one_out = leave_one_out
        self.window_pred = window_pred
        self.hrtf = hrtf
        self.norm_hrtf_diff = norm_hrtf_diff
        self.eeg_band = eeg_band
        self.spatial_locus = spatial_locus
        self.stim_input = stim_input
        self.cv_fold = cv_fold
        self.n_fold = 5 if dataset == 'fulsang' else 4
        self.trial_len = get_trials_len(dataset)
        self.dataset_n_trials = get_n_trials(dataset)
        self.eeg_chan = get_channels(dataset)
        self.stim_chan = 2 if self.hrtf else 1
        self.seed = seed

        if not isinstance(subjects, list):
            subjects = [subjects]

        if dataset != 'skl':
            
            # SUBJECT SPECIFIC
            if not leave_one_out:
                
                self.subjects = subjects
                self.trials = get_trials(self.split, self.dataset_n_trials, self.cv_fold, shuffle=self.rnd_trials, fixed=False, dataset= self.dataset)

            # SUBJECT INDEPENDENT
            else:
                if split == 'test':
                    self.subjects = subjects

                else:
                    train_val_subjects = get_other_subjects(subjects, dataset)
                    # Pick randomly 4 subjects from the list for each of the 5 CV fold when val or training
                    random.seed(self.seed)
                    val_cv_subjects = [random.sample(train_val_subjects, 4) for _ in range(self.n_fold)]

                    if cv_fold is not None:
                        val_subjects = val_cv_subjects[cv_fold]
                    else:
                        val_subjects = val_cv_subjects[0]

                    if split=='train':
                        self.subjects = [subj for subj in train_val_subjects if subj not in val_subjects]
                    elif split=='val':
                        self.subjects = val_subjects
                    else:
                        raise ValueError('Introduce a valid set')
                    
                self.trials = get_trials('all',  self.dataset_n_trials, shuffle=self.rnd_trials, fixed=self.fixed, dataset= self.dataset)

            self.n_trials = len(self.trials)

        else: self.subjects = subjects

        if dataset == 'fulsang':
            self.eeg, self.stima, self.stimb = self.get_Fulsang_data()
        elif dataset == 'jaulab' or dataset == 'kuleuven':
            self.eeg, self.stima, self.stimb = self.get_Jaulab_KULeuven_data()
        elif dataset == 'skl':
            self.eeg, self.stima = self.get_SKL_data()
            self.stimb = None # No unattended stim on SparKuLee experiment
        else:
            raise ValueError('Introduce a valid dataset name between fulsang/kuleuven/jaulab/skl')
        
        # STIM INPUT
        if stim_input:

            # Generate the corresponding labels (1 attended 0 non attended)
            self.labels = torch.cat((torch.ones_like(self.stima), torch.zeros_like(self.stima)), dim=1)

            # Duplicate the data with different stim order
            self.eeg = torch.cat((self.eeg, self.eeg), dim=1)
            stima, stimb = self.stima.clone(), self.stimb.clone()
            self.stima = torch.cat((stima, stimb), dim=1)
            self.stimb = torch.cat((stimb, stima), dim=1)            

        self.window = window
        self.n_samples = self.eeg.shape[1]

    def get_Fulsang_data(self):
        
        # Data matrices with shapes (subj, trial, T, C)
        eeg = torch.zeros((len(self.subjects), self.n_trials, self.trial_len, self.eeg_chan))
        stima = torch.zeros((len(self.subjects), self.n_trials, self.trial_len, self.stim_chan))
        stimb = torch.zeros((len(self.subjects), self.n_trials, self.trial_len, self.stim_chan))
        stima_diff = torch.zeros((len(self.subjects), self.n_trials, self.trial_len, 1)) if self.hrtf else None
        stimb_diff = torch.zeros((len(self.subjects), self.n_trials, self.trial_len, 1)) if self.hrtf else None

        for n, subject in enumerate(self.subjects):

            if self.data_type == 'mat': 

                # Load the eeg data
                preproc_data = scipy.io.loadmat(os.path.join(self.data_path ,subject + '_data_preproc.mat'))
                self.chan_idx = preproc_data['data']['dim'][0,0][0,0]['chan']['eeg'][0,0][0,0]
                self.chan_idx = np.array([chan[0] for chan in self.chan_idx[0]])
                
                # Load hrtf data where 2 stim channels are provided
                if self.hrtf:
                    
                    # Load HRTF stim data, the folder containing HRTF info. must be on the parent data_path dir (if not specify it)
                    parent_dir = os.path.dirname(self.data_path)
                    hrtfs_path = os.path.join(parent_dir, 'HRTFs')
                    hrtfs_data = scipy.io.loadmat(os.path.join(hrtfs_path ,subject + '_hrtfs.mat'))

                    for t, trial in enumerate(self.trials):

                        eeg[n, t] = torch.tensor(preproc_data['data']['eeg'][0,0][0,trial][:, :64]) # Only select the first 64 channels
                        stima[n, t] = torch.tensor(hrtfs_data['wavs'][:,0][trial]['wavA'][0, 0])
                        stimb[n, t] = torch.tensor(hrtfs_data['wavs'][:,0][trial]['wavB'][0, 0])

                        # HRTFs channel difference
                        stima_diff[n, t] = (stima[n, t, :, 0] - stima[n, t, :, 1]).unsqueeze(1)
                        stimb_diff[n, t] = (stimb[n, t, :, 0] - stimb[n, t, :, 1]).unsqueeze(1)

                        # Scale the channels according to the normalization between -1 and 1 of the differences
                        if self.norm_hrtf_diff: 

                            # Fit scaler
                            scaler = MinMaxScaler((-1, 1))
                            stim_cat = torch.cat((stima_diff[n, t], stimb_diff[n, t]))
                            norm_stim_cat = torch.tensor(scaler.fit_transform(stim_cat))

                            # Scale the original channels
                            stim_cat = torch.cat((stima[n, t, :, 0], stimb[n, t, :, 0]))
                            stim_cat = torch.cat((stima[n, t, :, 0], stimb[n, t, :, 0]))
                            stima[n, t, :, 0], stima[n, t, :, 1] = torch.tensor(scaler.transform(stima[n, t, :, 0].unsqueeze(1)))[:, 0], torch.tensor(scaler.transform(stima[n, t, :, 1].unsqueeze(1)))[:, 0]
                            stimb[n, t, :, 0], stimb[n, t, :, 1] = torch.tensor(scaler.transform(stimb[n, t, :, 0].unsqueeze(1)))[:, 0], torch.tensor(scaler.transform(stimb[n, t, :, 1].unsqueeze(1)))[:, 0]
                            stima_diff[n, t], stimb_diff[n, t] = norm_stim_cat[:self.trial_len], norm_stim_cat[self.trial_len:]
                
                # Load spatial data: 0 or 1 depending on the attended direction
                elif self.spatial_locus:
                    
                    # Load the attended directions from the expinfo.csv that it's located on a forder named Expinfo on the parent folder
                    parent_dir = os.path.dirname(self.data_path)
                    expinfo_path = os.path.join(parent_dir, 'Expinfo')
                    expinfo_path = os.path.join(expinfo_path ,'expinfo_' + subject + '.csv')
                    expinfo = pd.read_csv(expinfo_path)
                    # Filter out the trials with a single speaker
                    expinfo = expinfo[expinfo['n_speakers'] == 2]

                    for t, trial in enumerate(self.trials):

                        eeg[n, t] = torch.tensor(preproc_data['data']['eeg'][0,0][0,trial][:, :64]) # Only select the first 64 channels
                        # attended_lr csv file with values 1 for left attention and 2 for right attention
                        attended_lr = expinfo.iloc[t].attend_lr
                        # If attended right insert ones if not the tensor alrady filled with 0s
                        if attended_lr == 2:
                            stima[n, t] = torch.ones((self.trial_len, 1))
                        else:
                            stimb[n, t] = torch.ones((self.trial_len, 1))
                    
                # Load mono stim data
                else:

                    for t, trial in enumerate(self.trials):

                        eeg[n, t] = torch.tensor(preproc_data['data']['eeg'][0,0][0,trial][:, :64]) # Only select the first 64 channels
                        stima[n, t] = torch.tensor(preproc_data['data']['wavA'][0,0][0,trial])
                        stimb[n, t] = torch.tensor(preproc_data['data']['wavB'][0,0][0,trial])
                
            elif self.data_type == 'npy':

                folder_name = 'eeg' if self.eeg_band is None else 'eeg_band_' + self.eeg_band
                eeg[n] = torch.tensor(np.load(os.path.join(self.data_path, folder_name, subject+'_eeg.npy'), allow_pickle=True)[self.trials]).permute(0, 2, 1)
                stima[n] = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stima.npy'), allow_pickle=True)[self.trials]).unsqueeze(2)
                stimb[n] = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stimb.npy'), allow_pickle=True)[self.trials]).unsqueeze(2)
                self.chan_idx = None

                # Aplanar el tensor por trials (chan, trials*samples) y (trials*samples) 
                # BUG: Hacerlo descomponiendo el vector y utilizando hstack en vez de view como para archivo .mat
                # eeg_data = eeg_data.view(eeg_data.shape[1], eeg_data.shape[0] * eeg_data.shape[2])
                # eeg[n] = torch.hstack([torch.tensor(eeg[n][trial]) for trial in range(self.n_trials)])
                # stima[n] = stima[n].view(-1)
                # stimb[n] = stimb[n].view(-1)

            else:
                raise ValueError('Data type value has to be npy or mat')

        # Return one big matrix with the concatenated info (subj * trial * T, C)
        return eeg.reshape(-1, self.eeg_chan).T, stima.reshape(-1, self.stim_chan).T, stimb.reshape(-1, self.stim_chan).T
        
    def get_SKL_data(self):
    
        eeg = []
        stim = []

        subj_idx = [get_SKL_subj_idx(subj) for subj in self.subjects]
        filelist = os.listdir(self.data_path)

        # Subject specific/population mode
        if not self.leave_one_out:
            for subj in subj_idx:
                for n, file in enumerate(filelist):
                    chunks = file.split('_')
                    # Cargo la información del sujeto dependiendo del split
                    if self.split == chunks[0] and subj == chunks[2]:
                        data = torch.tensor(np.load(os.path.join(self.data_path, file)))
                        if 'eeg' in chunks[-1]:
                            eeg.append(data)
                        elif 'envelope' in chunks[-1]:
                            stim.append(data)

        # Subject independent mode
        else:
            for subj in subj_idx:
                # Carga por lotes para la operación torch.cat: ayuda al rendimiento en especial al entrenar
                print(f'Gathering data from subject {subj} on {self.split} loader')
                eeg_subj = []
                stim_subj = []
                for n, file in enumerate(filelist):
                    chunks = file.split('_')
                    # Cargo la información del sujeto (todos los splits)
                    if subj == chunks[2]:
                        data = torch.tensor(np.load(os.path.join(self.data_path, file)))
                        if 'eeg' in chunks[-1]:
                            eeg_subj.append(data)
                        elif 'envelope' in chunks[-1]:
                            stim_subj.append(data)
                cat_eeg = torch.cat(eeg_subj, dim=0)
                cat_stim = torch.cat(stim_subj, dim=0)
                half_samples = cat_eeg.shape[0] // 2
                # Totalidad de muestras para el entrenamiento y carga por lotes
                if self.split =='train':
                    eeg.append(cat_eeg)
                    stim.append(cat_stim)
                    gc.collect()
                # La mitad de muestras para val y train del sujeto excluido en el entrenamiento
                elif self.split =='val':
                    eeg.append(cat_eeg[:half_samples, :])
                    stim.append(cat_stim[:half_samples, :])
                elif self.split =='test':
                    eeg.append(cat_eeg[half_samples:, :])
                    stim.append(cat_stim[half_samples:, :])
                else: raise ValueError('Introduce a valid split name between train val or test')
        
        eeg_cat = torch.cat(eeg).T
        stima_cat = torch.cat(stim).T
        
        return eeg_cat, stima_cat
    
    def get_Jaulab_KULeuven_data(self):

        eeg = []
        stima = []
        stimb = []

        for subject in self.subjects:

            if self.data_type == 'mat': 
                preproc_data = scipy.io.loadmat(os.path.join(self.data_path ,subject + '_preproc.mat'))
                eeg_data = preproc_data['data']['eeg'][0,0][0,self.trials]
                stima_data = preproc_data['data']['wavA'][0,0][0,self.trials]
                stimb_data = preproc_data['data']['wavB'][0,0][0,self.trials]

                eeg_data = torch.hstack([normalize_eeg(torch.tensor(eeg_data[trial]).T) for trial in range(self.n_trials)])
                stima_data = torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(self.n_trials)]))
                stimb_data = torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(self.n_trials)]))
                
                # For zeroed eeg channels (48th channel) on KULeuven NaN values generated when tensor conversion
                if eeg_data.isnan().any():
                    eeg_data = torch.nan_to_num(eeg_data, nan=0.0)

            elif self.data_type == 'npy':
                eeg_data = np.load(os.path.join(self.data_path, 'eeg', subject+'_eeg.npy'), allow_pickle=True)[self.trials]
                stima_data = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stima.npy'), allow_pickle=True)[self.trials])
                stimb_data = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stimb.npy'), allow_pickle=True)[self.trials])

                # Aplanar el tensor por trials (chan, trials*samples) y (trials*samples) 
                eeg_data = torch.hstack([torch.tensor(eeg_data[trial]) for trial in range(self.n_trials)])
                stima_data = stima_data.view(-1)
                stimb_data = stimb_data.view(-1)

            else:
                raise ValueError('Data type value has to be npy or mat')

            eeg.append(eeg_data)
            stima.append(stima_data)
            stimb.append(stimb_data)

        # Concateno en un tensor global la información de los SUJETOS INDICADOS
        return torch.hstack(eeg), torch.unsqueeze(torch.cat(stima),0), torch.unsqueeze(torch.cat(stimb),0)
    
    def __len__(self):
        return (self.n_samples - self.window) // self.hop

    def __getitem__(self, idx):

        start = idx * self.hop
        end = start + self.window

        eeg = self.eeg[:, start:end] 
        stima = self.stima[:, start] if not self.window_pred else self.stima[0, start:end]
        if self.dataset != 'skl':
            stimb = self.stimb[:, start] if not self.window_pred else self.stimb[0, start:end]
            if self.stim_input:
                # Get the stimulus idx label from the starting point
                labels = self.labels[:, start]
                return {'eeg':eeg, 'stima':stima, 'stimb':stimb, 'labels':labels}
            else:
                return {'eeg':eeg, 'stima':stima, 'stimb':stimb}
        else:
            return {'eeg':eeg, 'stima':stima}

class CustomPathDataset(Dataset):

    def __init__(self, dataset, data_path, split, subjects, transform, window, hop, leave_one_out, seed = 42):

        if dataset not in ['skl']:
            raise ValueError(f'Dataset {dataset} not supported')
        if split not in ['train', 'val', 'test', 'train_and_val']:
            raise ValueError(f'Split {split} not supported')
        if not isinstance(subjects, list):
            subjects = [subjects]
        
        self.dataset = dataset
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.window = window
        self.hop = hop
        self.leave_one_out = leave_one_out
        self.seed = seed
        
        filelist = os.listdir(self.data_path)
        self.eeg_paths = []

        if not leave_one_out:
            self.subjects = subjects
        else:
            if split == 'test':
                self.subjects = subjects
            else:
                train_val_subjects = get_other_subjects(subjects, dataset)
                # Pick randomly 10 subjects from the list for validation set when LOSO val
                random.seed(self.seed)
                val_subjects = random.sample(train_val_subjects, 10)
                if split=='train':
                    self.subjects = [subj for subj in train_val_subjects if subj not in val_subjects]
                elif split=='val':
                    self.subjects = val_subjects
                
        # Load all the paths
        for subj in self.subjects:
            subj_idx = get_SKL_subj_idx(subj)
            for n, file in enumerate(filelist):
                chunks = file.split('_')
                # Cargo la información del sujeto dependiendo del split
                if split == chunks[0] and subj_idx == chunks[2]:
                    if 'eeg' in chunks[-1]:
                        self.eeg_paths.append(file)

        print(f'Found {len(self.eeg_paths)} paths for {self.split} with subjects {self.subjects}')

    def __len__(self):
        return len(self.eeg_paths)
    
    def __getitem__(self, idx):

        eeg_path = os.path.join(self.data_path, self.eeg_paths[idx])
        audio_path = os.path.join(self.data_path, self.eeg_paths[idx].replace('eeg.npy', 'envelope.npy'))

        eeg = np.load(eeg_path)
        try:
            audio = np.load(audio_path)
        except:
            print(f'Could not find envelope for {self.eeg_paths[idx]}')
            return None
        
        if self.transform == 'normalize':
            # Standarize EEG and envelope on channel dimension
            eeg = (eeg - np.mean(eeg, axis=0, keepdims=True)) / np.std(eeg, axis=0, keepdims=True)
            audio = (audio - np.mean(audio, axis=0, keepdims=True)) / np.std(audio, axis=0, keepdims=True)

        windowed_eeg = self.window_data(eeg)
        windowed_envelope = self.window_data(audio)

        # Select n random windows with n=batch_size
        random_idx = np.random.randint(0, windowed_eeg.shape[0], 1)
        windowed_eeg = windowed_eeg[random_idx]
        windowed_envelope  = windowed_envelope[random_idx]

        windowed_eeg = torch.from_numpy(windowed_eeg).float().transpose(1, 2).squeeze(0)
        windowed_envelope = torch.from_numpy(windowed_envelope).float().transpose(1, 2).squeeze()

        return {'eeg':windowed_eeg, 'stima':windowed_envelope}
    
    def window_data(self, data):

        len_data = data.shape[0]
        n_channels = data.shape[1]
        n_windows = (len_data - self.window) // self.hop

        windowed_data = np.empty((n_windows, self.window, n_channels))
        for idx in range(n_windows):

            start = idx * self.hop
            end = start + self.window

            windowed_data[idx] = data[start:end, :]

        return windowed_data