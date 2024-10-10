import torch
from models.ridge import Ridge
from utils.functional import get_trials, get_filename, correlation, check_jaulab_chan
import numpy as np
import os
import pickle
from utils.datasets import CustomDataset
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

global_path = 'C:/Users/jaulab/Desktop/AAD/IA_poster'
key = 'population'

dataset = 'fulsang'
window_list = [64]
mdl_load_folder = os.path.join(global_path, 'Results', key, 'models', dataset+'_data')
models = os.listdir(mdl_load_folder)
models = [model for model in models if 'Ridge' in model] # filter ridge models

for window in window_list:
 
    for model in models:

        block_size = window
        filt = True if 'filt' in model else False
        dataset = dataset
        time_shift = 100
        set_decode_acc = True
        population = False
        fixed = False
        rnd_trials = False

        if fixed: assert(dataset=='jaulab') # Only fixed subject for jaulab dataset
        dataset_name = dataset+'_fixed' if fixed else dataset

        # DEFINE THE SAVE PATH
        dst_save_path = os.path.join(global_path, 'Results', key, 'eval_metrics', dataset_name+'_data', model)
        decAcc_save_path = os.path.join(global_path, 'Results', key, 'decode_accuracy', dataset_name+'_data', model)

        eval_results = {}
        nd_results = {} # construct a null distribution when evaluating
        dec_results = []

        selected_subjects = subjects[dataset+'_subj']

        for subj in selected_subjects:

            print(f'Evaluating Ridge on window {window//64}s with {dataset_name} dataset for subj {subj}')

            # GET THE MODEL PATH
            mdl_load_path = os.path.join(mdl_load_folder, model)
            if key == 'population':
                mdl_filename = os.listdir(mdl_load_path)[0]
            else:
                mdl_filename = get_filename(mdl_load_path, subj)

            # CARGA EL MODELO
            mdl = pickle.load(open(os.path.join(mdl_load_path, mdl_filename), 'rb'))

            # LOAD THE DATA
            test_set = CustomDataset(dataset, paths[dataset+'_path'], 'test', subj, window=block_size, hop=block_size, acc=set_decode_acc, filt=filt, 
                                     filt_path=paths[dataset+'_filt_path'], population=population, fixed=fixed, rnd_trials=rnd_trials)
            test_eeg, test_stima, test_stimb = test_set.eeg, test_set.stima, test_set.stimb
            test_stim_nd = torch.roll(test_stima.clone().detach(), time_shift)

            # EVALÚA EN FUNCIÓN DEL MEJOR ALPHA/MODELO OBTENIDO
            scores_a = mdl.score_in_batches(test_eeg.T, test_stima[:, np.newaxis], batch_size=block_size)
            scores_b = mdl.score_in_batches(test_eeg.T, test_stimb[:, np.newaxis], batch_size=block_size)
            scores_nd = mdl.score_in_batches(test_eeg.T, test_stim_nd[:, np.newaxis], batch_size=block_size) # ya selecciona el best alpha solo
            
            att_corr = 0
            for i in range(len(scores_a)):
                score_a = scores_a[i]
                score_b = scores_b[i]

                if score_a > score_b:
                    att_corr += 1

            eval_results[subj] = [score for score in np.squeeze(scores_a)]
            nd_results[subj] = [score for score in np.squeeze(scores_nd)]
            dec_accuracy = (att_corr / len(scores_a)) * 100
            dec_results.append(dec_accuracy)

            if set_decode_acc:
                print(f'Subject {subj} | corr_mean {np.mean(scores_a):.4f} | decode_accuracy {dec_accuracy}')
            else:
                print(f'Subject {subj} | corr_mean {np.mean(scores_a):.4f}')

        str_win = str(block_size//64)+'s'

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