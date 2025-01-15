import torch
import torch.nn.functional as F
from utils.functional import multiple_loss_opt, set_seeds, get_data_path, get_channels, get_subjects, get_filename, get_loss
from utils.datasets import CustomDataset
from utils.loss_functions import CustomLoss
from torch.utils.data import DataLoader
from models.dnn import FCNN, CNN
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from statistics import mean
import numpy as np
import argparse
import yaml
import os
import json
import wandb
import sys

def main(config, wandb_upload, dataset, key, finetuned):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'spatial_audio'
    # wandb_upload=True
    # window_list = [64, 128, 320, 640, 1280, 2560] # 1s, 2s, 5s, 10s, 20s, 40s
    window_list = [64, 128, 320, 640, 1600, 3200] # 1s, 2s, 5s, 10s, 25s, 50s
    # window_list = [3200]
    
    # REPRODUCIBILITY
    if 'seed' in config.keys(): 
        set_seeds(config['seed'])
        exp_name =  exp_name + '_' + config['seed']
    else: 
        set_seeds() # default seed = 42
    
    for run in config['runs']:

        # Global params
        model = run['model']
        exp_name = config['exp_name'] + '_' + model

        mdl_load_folder = 'finetune_models' if finetuned else 'models'

        mdl_load_path = os.path.join(global_path, 'results', project, key, mdl_load_folder)

        tag = 'evaluation_finetune' if finetuned else 'evaluation'

        if wandb_upload: wandb.init(project=project, name=exp_name, tags=[tag], config=run)

        window_accuracies = {win//64: None for win in window_list}

        for eval_window in window_list:

            # Load config
            ds_config = run['dataset_params']
            train_params = run['train_params']
            
            window_pred = ds_config['window_pred'] if 'window_pred' in ds_config.keys() else not ds_config['unit_output']
            preproc_mode = ds_config['preproc_mode'] if 'preproc_mode' in ds_config.keys() else None
            data_path = get_data_path(global_data_path, dataset, preproc_mode = preproc_mode)
            window_len = ds_config['window_len'] if not window_pred else eval_window
            hop = ds_config['hop'] if window_pred else 1
            leave_one_out = True if key == 'subj_independent' else False
            data_type = ds_config['data_type'] if 'data_type' in ds_config.keys() else 'mat'
            eeg_band = ds_config['eeg_band'] if 'eeg_band' in ds_config.keys() else None
            fixed = ds_config['fixed']
            rnd_trials = ds_config['rnd_trials']
            hrtf = ds_config['hrtf'] if 'hrtf' in ds_config.keys() else False
            norm_hrtf_diff = ds_config['norm_hrtf_diff'] if 'norm_hrtf_diff' in ds_config.keys() else False
            spatial_locus = ds_config['spatial_locus'] if 'spatial_locus' in ds_config.keys() else False
            time_shift = 100
            dec_acc = True if dataset != 'skl' else False # skl dataset without unattended stim => dec-acc is not possible
            batch_size =  eval_window if not window_pred else batch_size
            lr = float(train_params['lr'])
            loss_mode = train_params['loss_mode'] if 'loss_mode' in train_params.keys() else 'mean'
            alpha = train_params['alpha_loss'] if 'alpha_loss' in train_params.keys() else 0

            # If the loss mode only takes ILD into account and is finetunned, eval with both correlation and ils with alpha=0.1
            if loss_mode in ['ild_mae', 'ild_mse'] and finetuned:
                loss_mode = 'corr_' + loss_mode
                alpha = 0.1

            if fixed: assert(dataset=='jaulab') # Only fixed subject for jaulab dataset
            dataset_name = dataset+'_fixed' if fixed else dataset

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # GET THE MODEL PATH
            mdl_name = f'{model}_batch={train_params["batch_size"]}_block={ds_config["window_len"]}_lr={lr}'

            # Add extensions to the model name depending on the params
            if preproc_mode is not None: mdl_name += '_' + preproc_mode
            if eeg_band is not None: mdl_name += '_' + eeg_band
            if loss_mode != 'mean': mdl_name += '_' + loss_mode
            if alpha != 0: mdl_name += '_alpha=' + str(alpha)
            if rnd_trials: mdl_name += '_rnd'
            if hrtf: mdl_name += '_hrtf'
            if norm_hrtf_diff: mdl_name += '_norm'

            if finetuned:
                eval_path, dec_path = 'eval_finetuned_metrics', 'decode_finetuned_accuracy'  
            else:
                eval_path, dec_path = 'eval_metrics', 'decode_accuracy'

            # DEFINE THE SAVE PATH
            dst_save_path = os.path.join(global_path, 'results', project, key, eval_path, dataset_name+'_data', mdl_name)
            decAcc_save_path = os.path.join(global_path, 'results', project, key, dec_path, dataset_name+'_data', mdl_name)

            eval_results = {}
            nd_results = {} # construct a null distribution when evaluating
            dec_results = []
            eval_mean_results = []
            
            selected_subjects = get_subjects(dataset)

            for subj in selected_subjects:

                print(f'Evaluating {model} on window {eval_window//64}s with {dataset_name} dataset for subj {subj}')

                if loss_mode is not None: print(f'Optimizing the network based on {loss_mode} criterion')

                mdl_folder = os.path.join(mdl_load_path, dataset_name+'_data', mdl_name)
                if key == 'population' and not finetuned:
                    mdl_filename = os.listdir(mdl_folder)[0] # only a single trained model
                else:
                    mdl_filename = get_filename(mdl_folder, subj) # search for the model related with the subj
                
                # LOAD THE MODEL
                if model == 'FCNN':
                    run['model_params']['n_chan'] = get_channels(dataset)
                    mdl = FCNN(**run['model_params'])
                elif model == 'CNN':
                    run['model_params']['input_channels'] = get_channels(dataset)
                    mdl = CNN(**run['model_params'])
                elif model == 'VLAAI':
                    run['model_params']['input_channels'] = get_channels(dataset)
                    mdl = VLAAI(**run['model_params'])
                elif model == 'Conformer':
                    run['model_params']['kernel_chan'] = get_channels(dataset)
                    run['model_params']['eeg_channels'] = get_channels(dataset)
                    mdl_config = ConformerConfig(**run['model_params'])
                    mdl = Conformer(mdl_config)
                else:
                    raise ValueError('Introduce a valid model')

                mdl.load_state_dict(torch.load(os.path.join(mdl_folder, mdl_filename), map_location=torch.device(device)))
                mdl.to(device)

                # LOAD THE DATA
                test_set = CustomDataset(dataset, data_path, 'test', subj, window=window_len, hop=hop, data_type=data_type, leave_one_out=leave_one_out, 
                                        fixed=fixed, rnd_trials = rnd_trials, window_pred=window_pred, hrtf=hrtf, norm_hrtf_diff=norm_hrtf_diff, eeg_band=eeg_band, 
                                        spatial_locus=spatial_locus)
                test_loader = DataLoader(test_set, batch_size, shuffle=window_pred, pin_memory=True)
                
                # LOSS FUNCTION
                criterion = CustomLoss(mode=loss_mode, window_pred=window_pred, alpha_end=alpha)

                # EVALUATE THE MODEL
                eval_loss = []
                eval_nd_loss = []
                eval_loss_list = []
                att_corr = 0

                with torch.no_grad():
                    for batch, data in enumerate(test_loader):
                        
                        eeg = data['eeg'].to(device, dtype=torch.float)
                        stima = data['stima'].to(device, dtype=torch.float)
                        
                        y_hat = mdl(eeg)

                        loss_list = criterion(preds=y_hat, targets = stima)
                        loss = loss_list[0]

                        # Calculates Pearson's coef. for the matching distribution and for the null one
                        nd_loss = criterion(preds=y_hat, targets = torch.roll(stima, time_shift))[0]

                        if dec_acc:
                            if loss_mode != 'spatial_locus':
                                stimb = data['stimb'].to(device, dtype=torch.float)
                                unat_loss_list = criterion(preds=y_hat, targets = stimb)
                                unat_loss = criterion(preds=y_hat, targets = stimb)[0]
                                # Decoding accuracy
                                if loss.item() < unat_loss.item():
                                    att_corr += 1

                        # Append all losses for eval results
                        eval_loss.append(loss.item())
                        eval_loss_list.append(loss_list)
                        eval_nd_loss.append(nd_loss.item())

                eval_results[subj] = eval_loss
                nd_results[subj] = eval_nd_loss
                dec_accuracy = (att_corr / len(test_loader)) * 100
                dec_results.append(dec_accuracy)
                
                if multiple_loss_opt(loss_mode):
                    eval_mean_results.append([mean(eval_loss), torch.mean(torch.hstack([loss_list[1] for loss_list in eval_loss_list])).item(), torch.mean(torch.hstack([loss_list[2] for loss_list in eval_loss_list])).item()])
                else:
                    eval_mean_results.append([mean(eval_loss)])

                print(f'Subject {subj} | corr_mean {mean(eval_loss)} | decode_accuracy {dec_accuracy}')

            if wandb_upload:
                loss_mean_results = [results[0] for results in eval_mean_results]
                wandb_log = {'window': eval_window, 'loss_subj_mean': np.mean(loss_mean_results), 'loss_subj_std': np.std(loss_mean_results), 'decAcc_subj_mean': np.mean(dec_results), 'decAcc_subj_std': np.std(dec_results)}
                # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
                if multiple_loss_opt(loss_mode):
                    wandb_log['corr_subj_mean'] = np.mean([results[1] for results in eval_mean_results])
                    wandb_log['ild_subj_mean'] = np.mean([results[2] for results in eval_mean_results])
                    wandb_log['corr_subj_std'] = np.std([results[1] for results in eval_mean_results])
                    wandb_log['ild_subj_std'] = np.std([results[2] for results in eval_mean_results])
                wandb.log(wandb_log)

            # Save the window results to compute mesd
            window_accuracies[eval_window//64] = dec_results

            str_win = str(eval_window//64)+'s' if 'VLAAI' in model else str(batch_size//64)+'s'
            
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

        # COMPUTE MESD (default values)
        print('Computing MESD')
        mesd_dict = {'mesd': [], 'N_mesd': [], 'tau_mesd': [], 'p_mesd': []}
        for subj in range(len(selected_subjects)):
            tau = np.array(list(window_accuracies.keys()))
            p = np.array([results[subj] / 100 for results in window_accuracies.values()])
            if np.any(p > 0.5):
                mesd, N_mesd, tau_mesd, p_mesd = compute_MESD(tau,p)
                print(f"For subject {subj} the minimal expected switch duration is MESD = {mesd} \nat an otpimal working point of (tau, p) = ({tau_mesd}, {p_mesd}) \nwith N = {N_mesd} states in the Markov chain.")
            else:
                mesd, N_mesd, tau_mesd, p_mesd = 0, 0, 0, 0
                print(f"For subject {subj} the MESD could not be computed")
            mesd_results = [mesd, N_mesd, tau_mesd, p_mesd]
            for idx, value in zip(list(mesd_dict.keys()), mesd_results):
                mesd_dict[idx].append(value)
            # Wandb upload with the specific info of the specific subject
            upload_data = {idx: value[subj] for idx, value in mesd_dict.items()}
            upload_data['subject'] = subj
            if wandb_upload: wandb.log(upload_data)
        if wandb_upload: wandb.log({'mesd_mean': np.mean(mesd_dict['mesd']), 'mesd_median': np.median(mesd_dict['mesd'])})
        json.dump(mesd_dict, open(os.path.join(decAcc_save_path, 'mesd'),'w'))

        if wandb_upload: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default='configs/spatial_audio/locus_best_models.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--finetuned", action='store_true', help="When included search for the model on the finetune folder")

    args = parser.parse_args()

    wandb_upload = args.wandb

    # Introduce path to the mesd-toolbox
    sys.path.append(r"C:\Users\jaulab\Desktop\AAD\mesd-toolbox\mesd-toolbox-python")
    from mesd_toolbox import compute_MESD
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload, args.dataset, args.key, args.finetuned)