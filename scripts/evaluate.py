import torch
import os
from utils.functional import get_trials, get_filename, correlation, check_jaulab_chan
from utils.datasets import CustomDataset
from torch.utils.data import DataLoader
from models.dnn import FCNN
from models.vlaai import VLAAI
from statistics import mean
import json

# Select subjects you want your network to be evaluated on
subjects = {
    'fulsang_subj': ['S'+str(n) for n in range(1, 19)],
    'skl_subj': ['S'+str(n) for n in range(1, 86)],
    'jaulab_subj' : ['S'+str(n) for n in range(1, 18)]
}

# Data path parameters
data_global_path = 'C:/Users/jaulab/Desktop/AAD/Data'
paths = {'hugo_path': data_global_path + "/Hugo_2022/hugo_preproc_data",
        'fulsang_path': data_global_path + '/Fulsang_2017/DATA_preproc',
        'jaulab_path': data_global_path + '/Jaulab_2024/PreprocData_ICA',
        'fulsang_filt_path': data_global_path + '/Fulsang_2017/DATA_filtered',
        'jaulab_filt_path': data_global_path + '/Jaulab_2024/DATA_filtered',
        'skl_path': data_global_path + '/SKL_2023/split_data',
        'skl_filt_path': None
}

channels = {
    'skl': 64,
    'fulsang': 64,
    'jaulab': 61
}

"""

    Evaluating parameters
    ------------------

    window_len: int
        number of samples for the window to predict the ongoing sample (block_size)

    batch_size: int
        batch_size for selecting the number of samples generated when evaluating

    batch_size: int
        batch size selected for the dataloader that group evaluating windows

    dataset: string
        name of the dataset you want to evaluate from 'fulsang', 'jaulab' or 'skl'

    time_shift: int
        number of samples for performing the circular time shift to obtain the 
        null distribution, common case between 1 and 2s (64 and 128 samples)

    set_decode_acc: bool
        if True the decoding accuracy is also obtained when the datasets are 'fulsang'
        or 'hugo'

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

global_path = 'C:/Users/jaulab/Desktop/AAD/IA_poster'
key = 'population'
global_mode = False

dataset = 'jaulab'
window_list = [64, 128, 320, 640, 1600]
mdl_load_folder = os.path.join(global_path, 'Results', key, 'models', dataset+'_data')
models = os.listdir(mdl_load_folder)
models = [model for model in models if 'Ridge' not in model] # not ridge
# models = [model for model in models if 'lr=1e-06' not in model] # neither lr=1e-06
# models = [model for model in models if 'FCNN' not in model]

for window in window_list:
 
    for model in models:

        block_size = window if 'VLAAI' in model else 50
        hop = window if 'VLAAI' in model else 1
        batch_size = 1 if 'VLAAI' in model else window
        filt = True if 'filt' in model else False
        dataset = dataset
        time_shift = 100
        set_decode_acc = True
        dropout = 0.4
        population = False
        fixed = False
        rnd_trials = False
        unit_output = False if 'VLAAI' in model else True
        shuffle = True if unit_output else False

        if fixed: assert(dataset=='jaulab') # Only fixed subject for jaulab dataset
        dataset_name = dataset+'_fixed' if fixed else dataset

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DEFINE THE SAVE PATH
        dst_save_path = os.path.join(global_path, 'Results', key, 'eval_metrics', dataset_name+'_data', model)
        decAcc_save_path = os.path.join(global_path, 'Results', key, 'decode_accuracy', dataset_name+'_data', model)

        eval_results = {}
        nd_results = {} # construct a null distribution when evaluating
        dec_results = []
        
        selected_subjects = subjects[dataset+'_subj']

        for subj in selected_subjects:

            print(f'Evaluating {model.split("_")[0]} on window {window//64}s with {dataset_name} dataset for subj {subj}')

            # GET THE MODEL PATH
            mdl_load_path = os.path.join(mdl_load_folder, model)
            if key == 'population':
                mdl_filename = os.listdir(mdl_load_path)[0]
            else:
                mdl_filename = get_filename(mdl_load_path, subj)

            # LOAD THE MODEL
            if 'FCNN' in model:
                mdl = FCNN(n_hidden=3, dropout_rate=dropout, n_chan=channels[dataset], 
                        n_samples = block_size)
            elif 'VLAAI' in model:
                mdl = VLAAI(n_blocks = 4, use_skip = True, input_channels=channels[dataset], 
                            output_dim=1, d = dropout)

            mdl.load_state_dict(torch.load(os.path.join(mdl_load_path, mdl_filename), map_location=torch.device(device)))
            mdl.to(device)

            # LOAD THE DATA
            test_set = CustomDataset(dataset, paths[dataset+'_path'], 'test', subj, window=block_size, hop=hop, acc=set_decode_acc, filt=filt, 
                                     filt_path=paths[dataset+'_filt_path'], population=population, fixed=fixed, rnd_trials=rnd_trials,
                                     unit_output=unit_output)
            test_loader = DataLoader(test_set, batch_size, shuffle=shuffle, pin_memory=True)
            
            # EVALUATE THE MODEL
            corr = []
            nd_corr = []
            att_corr = 0

            with torch.no_grad():
                for batch, data in enumerate(test_loader):
                    
                    eeg = data['eeg'].to(device, dtype=torch.float)
                    stima = data['stima'].to(device, dtype=torch.float)
            
                    y_hat, loss = mdl(eeg)

                    # Calculates Pearson's coef. for the matching distribution and for the null one
                    nd_acc = correlation(torch.roll(stima, time_shift), y_hat, batch_dim=unit_output)
                    acc = correlation(stima, y_hat, batch_dim=unit_output)

                    if set_decode_acc:
                        stimb = data['stimb'].to(device, dtype=torch.float)
                        unat_acc = correlation(stimb, y_hat, batch_dim=unit_output)
                        # loop on the batch size
                        if acc.item() > unat_acc.item():
                            att_corr += 1

                    corr.append(acc.item())
                    nd_corr.append(nd_acc.item())

            eval_results[subj] = corr
            nd_results[subj] = nd_corr
            dec_accuracy = (att_corr / len(test_loader)) * 100
            dec_results.append(dec_accuracy)

            if set_decode_acc:
                print(f'Subject {subj} | corr_mean {mean(corr)} | decode_accuracy {dec_accuracy}')
            else:
                print(f'Subject {subj} | corr_mean {mean(corr)}')

        str_win = str(block_size//64)+'s' if 'VLAAI' in model else str(batch_size//64)+'s'

        # SAVE RESULTS
        if not os.path.exists(dst_save_path):
            os.makedirs(dst_save_path)
        filename = str_win+'_Results'
        json.dump(eval_results, open(os.path.join(dst_save_path, filename),'w'))
        filename = str_win+'_nd_Results'
        json.dump(nd_results, open(os.path.join(dst_save_path, filename),'w'))

        # SAVE ACCURACY RESULTS
        if set_decode_acc:
            if not os.path.exists(decAcc_save_path):
                os.makedirs(decAcc_save_path)
            filename = str_win+'_accuracies'
            json.dump(dec_results, open(os.path.join(decAcc_save_path, filename),'w'))