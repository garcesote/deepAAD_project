import os
import scipy
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.functional import get_other_subjects, get_trials, get_leave_one_out_trials, normalize_eeg, get_data_path, get_SKL_subj_idx
import gc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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
                 norm_hrtf_diff=False, eeg_band= None, spatial_locus=False):

        if not isinstance(subjects, list):
            subjects = [subjects]

        if not leave_one_out:
            self.subjects = subjects
        else:
            # When training on population mode all subjects except the specified ore used for training
            if split=='train':
                self.subjects = get_other_subjects(subjects, dataset)
            # For val and test the specified subject is intraoduced
            else:
                self.subjects = subjects

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
        self.cv_fold = cv_fold

        if dataset == 'fulsang':
            self.eeg, self.stima, self.stimb = self.get_Fulsang_data()
        elif dataset == 'jaulab':
            self.eeg, self.stima, self.stimb = self.get_Jaulab_data()
        elif dataset == 'skl':
            self.eeg, self.stima = self.get_SKL_data()
            self.stimb = None # No unattended stim on SparKuLee experiment
        else:
            raise ValueError('Introduce a valid dataset name between fulsang or skl')

        self.window = window
        self.n_samples = self.eeg.shape[1]

    def get_Fulsang_data(self):
        
        # Fulsang dataset with 60 trials of 50s with 64 desired channels
        n_trials = 60
        trial_len = 3200
        eeg_chan = 64
        stim_chan = 2 if self.hrtf else 1
        
        if not self.leave_one_out:
            trials = get_trials(self.split, n_trials, self.cv_fold, shuffle=self.rnd_trials, fixed=False, dataset= self.dataset)
        else:
            trials = get_leave_one_out_trials(self.split, n_trials, alternate=self.rnd_trials, fixed=False)

        self.trials = trials
        self.n_trials = len(trials)
        
        # Data matrices with shapes (subj, trial, T, C)
        eeg = torch.zeros((len(self.subjects), self.n_trials, trial_len, eeg_chan))
        stima = torch.zeros((len(self.subjects), self.n_trials, trial_len, stim_chan))
        stimb = torch.zeros((len(self.subjects), self.n_trials, trial_len, stim_chan))
        stima_diff = torch.zeros((len(self.subjects), self.n_trials, trial_len, 1)) if self.hrtf else None
        stimb_diff = torch.zeros((len(self.subjects), self.n_trials, trial_len, 1)) if self.hrtf else None

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

                    for t, trial in enumerate(trials):

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
                            stima_diff[n, t], stimb_diff[n, t] = norm_stim_cat[:trial_len], norm_stim_cat[trial_len:]
                
                # Load spatial data: 0 or 1 depending on the attended direction
                elif self.spatial_locus:
                    
                    # Load the attended directions from the expinfo.csv that it's located on a forder named Expinfo on the parent folder
                    parent_dir = os.path.dirname(self.data_path)
                    expinfo_path = os.path.join(parent_dir, 'Expinfo')
                    expinfo_path = os.path.join(expinfo_path ,'expinfo_' + subject + '.csv')
                    expinfo = pd.read_csv(expinfo_path)
                    # Filter out the trials with a single speaker
                    expinfo = expinfo[expinfo['n_speakers'] == 2]

                    for t, trial in enumerate(trials):

                        eeg[n, t] = torch.tensor(preproc_data['data']['eeg'][0,0][0,trial][:, :64]) # Only select the first 64 channels
                        # attended_lr csv file with values 1 for left attention and 2 for right attention
                        attended_lr = expinfo.iloc[t].attend_lr
                        # If attended right insert ones if not the tensor alrady filled with 0s
                        if attended_lr == 2:
                            stima[n, t] = torch.ones((trial_len, 1))
                        else:
                            stimb[n, t] = torch.ones((trial_len, 1))
                    
                # Load mono stim data
                else:

                    for t, trial in enumerate(trials):

                        eeg[n, t] = torch.tensor(preproc_data['data']['eeg'][0,0][0,trial][:, :64]) # Only select the first 64 channels
                        stima[n, t] = torch.tensor(preproc_data['data']['wavA'][0,0][0,trial])
                        stimb[n, t] = torch.tensor(preproc_data['data']['wavB'][0,0][0,trial])
                
            elif self.data_type == 'npy':

                folder_name = 'eeg' if self.eeg_band is None else 'eeg_band_' + self.eeg_band
                eeg[n] = torch.tensor(np.load(os.path.join(self.data_path, folder_name, subject+'_eeg.npy'), allow_pickle=True)[trials]).permute(0, 2, 1)
                stima[n] = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stima.npy'), allow_pickle=True)[trials]).unsqueeze(2)
                stimb[n] = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stimb.npy'), allow_pickle=True)[trials]).unsqueeze(2)
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
        return eeg.reshape(-1, eeg_chan).T, stima.reshape(-1, stim_chan).T, stimb.reshape(-1, stim_chan).T
        
    def get_SKL_data(self):
    
        eeg = []
        stim = []
        gpu = True

        subj_idx = [get_SKL_subj_idx(subj) for subj in self.subjects]
        filelist = os.listdir(self.data_path)
        n_files = len(filelist)

        # Subject specific mode
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

        # Subject independent / population mode
        else:
            for subj in subj_idx:
                # Carga por lotes para la operación torch.cat: ayuda al rendimiento en especial al entrenar
                print(f'Gathering data from subject {subj} on {self.split} loader')
                eeg_subj = []
                stim_subj = []
                for n, file in enumerate(filelist):
                    chunks = file.split('_')
                    n_subj_files = len([file for file in filelist if subj in file]) # get number of files of each subject
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
    
    def get_Jaulab_data(self):

        eeg = []
        stima = []
        stimb = []

        n_trials = 96 # 96 trials of 26s per subject in Fulsang dataset
        if not self.leave_one_out:
            trials = get_trials(self.split, n_trials, shuffle=self.rnd_trials, fixed=self.fixed, dataset= self.dataset)
        else:
            trials = get_leave_one_out_trials(self.split, n_trials, alternate=self.rnd_trials, fixed=self.fixed)

        self.trials = trials
        self.n_trials = len(trials)

        for subject in self.subjects:

            if self.data_type == 'mat': 
                preproc_data = scipy.io.loadmat(os.path.join(self.data_path ,subject + '_preproc.mat'))
                eeg_data = preproc_data['data']['eeg'][0,0][0,trials]
                stima_data = preproc_data['data']['wavA'][0,0][0,trials]
                stimb_data = preproc_data['data']['wavB'][0,0][0,trials]

                eeg_data = torch.hstack([normalize_eeg(torch.tensor(eeg_data[trial]).T) for trial in range(self.n_trials)])
                stima_data = torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(self.n_trials)]))
                stimb_data = torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(self.n_trials)]))
                
            elif self.data_type == 'npy':
                eeg_data = np.load(os.path.join(self.data_path, 'eeg', subject+'_eeg.npy'), allow_pickle=True)[trials]
                stima_data = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stima.npy'), allow_pickle=True)[trials])
                stimb_data = torch.tensor(np.load(os.path.join(self.data_path, 'stim', subject+'_stimb.npy'), allow_pickle=True)[trials])

                # Aplanar el tensor por trials (chan, trials*samples) y (trials*samples) 
                eeg_data = torch.hstack([torch.tensor(eeg_data[trial]) for trial in range(self.n_trials)])
                stima_data = stima_data.view(-1)
                stimb_data = stimb_data.view(-1)

            else:
                raise ValueError('Data type value has to be npy or mat')

            # Añadir canales con zero padding si estos no llegan a 61: puede haber sujetos con 63, 62 o 61 electrodos
            n_channels = eeg_data.shape[0]
            rest_channels = 61 - n_channels
            zero_channels = torch.zeros((rest_channels, eeg_data.shape[1]))
            # zero_eeg = [torch.cat((eeg_data[trial], zero_channels), dim=0) for trial in range(self.n_trial)]
            zero_eeg = torch.cat((eeg_data, zero_channels), dim=0)

            eeg.append(zero_eeg)
            stima.append(stima_data)
            stimb.append(stimb_data)

        # Concateno en un tensor global la información de los SUJETOS INDICADOS
        return torch.hstack(eeg), torch.cat(stima), torch.cat(stimb)
    
    def __len__(self):
        return (self.n_samples - self.window) // self.hop

    def __getitem__(self, idx):

        start = idx * self.hop
        end = start + self.window

        eeg = self.eeg[:, start:end] 
        stima = self.stima[:, start] if not self.window_pred else self.stima[0, start:end]
        if self.dataset != 'skl':
            stimb = self.stimb[:, start] if not self.window_pred else self.stimb[0, start:end]
            return {'eeg':eeg, 'stima':stima, 'stimb':stimb}
        else:
            return {'eeg':eeg, 'stima':stima}
