import torch
from utils.functional import get_filename, get_subjects, str2bool, get_data_path
import numpy as np
import os
import pickle
from utils.datasets import CustomDataset
import json
import argparse
import wandb

def main(
        dataset: str,
        key: str,
        filt: bool,
        fixed: bool,
        rnd_trials: bool,
        wandb_upload: bool
    ):

    # Data path parameters
    global_data_path = 'C:/Users/jaulab/Desktop/AAD/Data'
    global_path = 'C:/Users/jaulab/Desktop/deepAAD_project'
    # global_path: 'C:/Users/garcia.127407/Desktop/DNN_AAD/deepAAD_project'
    # global_data_path: 'D:\igarcia\AAD_Data'
    project = 'replicate_model_results'
    
    data_path = get_data_path(global_data_path, dataset, filt=False)
    data_filt_path = get_data_path(global_data_path, dataset, filt=True) if filt else None

    window_list = [64, 128, 320, 640, 1600]
    time_shift = 100
    leave_one_out = True if key == 'subj_independent' else False # Attention! This parameter change the whole sim. (read doc)
    dec_acc = True if dataset != 'skl' else False # skl dataset without unattended stim => dec-acc is not possible
    
    if fixed: assert(dataset == 'jaulab')
    dataset_name = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

    # SAVE THE MODEL
    model = 'Ridge'
    # Add extensions to the model name depending on the params
    if filt:
        model = model + '_filt'
    if rnd_trials:
        model = model + '_rnd'

    # DEFINE THE SAVE PATH
    dst_save_path = os.path.join(global_path, 'Results', key, 'eval_metrics', dataset_name+'_data', model)
    decAcc_save_path = os.path.join(global_path, 'Results', key, 'decode_accuracy', dataset_name+'_data', model)
    mdl_load_folder = os.path.join(global_path, 'Results', key, 'models', dataset+'_data')
    
    exp_name = ('_').join([key, dataset_name, model])
    if wandb_upload: wandb.init(project=project, name=exp_name, tags=['evaluation_ridge'])

    for window in window_list:

        block_size = window
        dataset = dataset

        eval_results = {}
        nd_results = {} # construct a null distribution when evaluating
        dec_results = []
        eval_mean_results = []

        selected_subjects = get_subjects(dataset)

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
            test_set = CustomDataset(dataset, data_path, 'test', subj, window=block_size, hop=block_size, filt=filt, 
                                    filt_path= data_filt_path, leave_one_out=leave_one_out, fixed=fixed, rnd_trials=rnd_trials)
            test_eeg, test_stima, test_stimb = test_set.eeg, test_set.stima, test_set.stimb
            test_stim_nd = torch.roll(test_stima.clone().detach(), time_shift)

            # EVALÚA EN FUNCIÓN DEL MEJOR ALPHA/MODELO OBTENIDO
            scores_a = mdl.score_in_batches(test_eeg.T, test_stima[:, np.newaxis], batch_size=block_size)
            scores_nd = mdl.score_in_batches(test_eeg.T, test_stim_nd[:, np.newaxis], batch_size=block_size) # ya selecciona el best alpha solo
            
            # DECODING ACCURACY
            att_corr = 0
            if dec_acc:
                scores_b = mdl.score_in_batches(test_eeg.T, test_stimb[:, np.newaxis], batch_size=block_size)
                for i in range(len(scores_a)):
                    score_a = scores_a[i]
                    score_b = scores_b[i]

                    if score_a > score_b:
                        att_corr += 1

            eval_results[subj] = [score for score in np.squeeze(scores_a)]
            nd_results[subj] = [score for score in np.squeeze(scores_nd)]
            dec_accuracy = (att_corr / len(scores_a)) * 100
            dec_results.append(dec_accuracy)
            eval_mean_results.append(np.mean(scores_a))

            print(f'Subject {subj} | corr_mean {np.mean(scores_a):.4f} | decode_accuracy {dec_accuracy}')

        if wandb_upload:
                wandb.log({'window': block_size, 'corr_subj_mean': np.mean(eval_mean_results), 'corr_subj_std': np.std(eval_mean_results), 'decAcc_subj_mean': np.mean(dec_results), 'decAcc_subj_std': np.std(dec_results)})

        str_win = str(block_size//64)+'s'

        # SAVE RESULTS
        if not os.path.exists(dst_save_path):
            os.makedirs(dst_save_path)
        filename = str_win+'_Results'
        json.dump(eval_results, open(os.path.join(dst_save_path, filename),'w'))
        filename = str_win+'_nd_Results'
        json.dump(nd_results, open(os.path.join(dst_save_path, filename),'w'))

        # SAVE ACCURACY RESULTS
        if not os.path.exists(decAcc_save_path):
            os.makedirs(decAcc_save_path)
        filename = str_win+'_accuracies'
        json.dump(dec_results, open(os.path.join(decAcc_save_path, filename),'w'))
    
    if wandb_upload: wandb.finish()
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval Ridge script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--filt", type=str2bool, default='False', help="EEG filtered")
    parser.add_argument("--fixed", type=str2bool, default='False', help="Static Jaulab trials")
    parser.add_argument("--rnd_trials", type=str2bool, default='False', help="Random trial selection")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")

    args = parser.parse_args()

    wandb_upload = args.wandb
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()
    
    # Llamar a la función de entrenamiento con los argumentos
    main(
        args.dataset,
        args.key,
        args.filt,
        args.fixed,
        args.rnd_trials,
        args.wandb
    )