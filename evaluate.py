import torch
from utils.functional import get_data_path, get_channels, get_subjects, get_filename, correlation
from utils.datasets import CustomDataset
from torch.utils.data import DataLoader
from models.dnn import FCNN
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from statistics import mean
import numpy as np
import argparse
import yaml
import os
import json
import wandb

def main(config, wandb_upload):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'replicate_model_results'
    window_list = [64, 128, 320, 640, 1600]
    
    for exp in config['experiments']:

        # Global params
        exp_name = exp['name']
        key = exp['key']
        dataset = exp['dataset']
        model = exp['model']

        mdl_save_path = global_path + '/results/'+exp['key']+'/models'

        if wandb_upload: wandb.init(project=project, name=exp_name, tags=['evaluation'], config=exp)

        for eval_window in window_list:

            # Load config
            ds_config = exp['dataset_params']
            train_params = exp['train_params']
            
            unit_output = ds_config['unit_output']
            data_path = get_data_path(global_data_path, dataset, filt=False)
            window_len = ds_config['window_len'] if unit_output else eval_window
            hop = ds_config['hop']
            leave_one_out = True if exp['key'] == 'subj_independent' else False
            filt = ds_config['filt']
            filt_path = get_data_path(global_data_path, dataset, filt=True) if filt else None
            fixed = ds_config['fixed']
            rnd_trials = ds_config['rnd_trials']
            time_shift = 100
            dec_acc = True if dataset != 'skl' else False # skl dataset without unattended stim => dec-acc is not possible

            batch_size =  eval_window if unit_output else 1
            lr = float(train_params['lr'])

            if fixed: assert(dataset=='jaulab') # Only fixed subject for jaulab dataset
            dataset_name = dataset+'_fixed' if fixed else dataset

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # GET THE MODEL PATH
            mdl_name = f'{model}_batch={train_params["batch_size"]}_block={ds_config["window_len"]}_lr={lr}'

            # Add extensions to the model name depending on the params
            if filt:
                mdl_name = mdl_name + '_filt'
            if rnd_trials:
                mdl_name = mdl_name + '_rnd'

            # DEFINE THE SAVE PATH
            dst_save_path = os.path.join(global_path, 'Results', key, 'eval_metrics', dataset_name+'_data', model)
            decAcc_save_path = os.path.join(global_path, 'Results', key, 'decode_accuracy', dataset_name+'_data', model)

            eval_results = {}
            nd_results = {} # construct a null distribution when evaluating
            dec_results = []
            eval_mean_results = []
            
            selected_subjects = get_subjects(dataset)

            for subj in selected_subjects:

                print(f'Evaluating {model} on window {eval_window//64}s with {dataset_name} dataset for subj {subj}')
            
                mdl_folder = os.path.join(mdl_save_path, dataset_name+'_data', mdl_name)
                if key == 'population':
                    mdl_filename = os.listdir(mdl_folder)[0] # only a single trained model
                else:
                    mdl_filename = get_filename(mdl_folder, subj) # search for the model related with the subj
                
                # LOAD THE MODEL
                if model == 'FCNN':
                    exp['model_params']['n_chan'] = get_channels(dataset)
                    mdl = FCNN(**exp['model_params'])
                elif model == 'VLAAI':
                    exp['model_params']['input_channels'] = get_channels(dataset)
                    mdl = VLAAI(**exp['model_params'])
                elif model == 'EEG_Conformer':
                    exp['model_params']['eeg_channels'] = get_channels(dataset)
                    mdl_config = ConformerConfig(**exp['model_params'])
                    mdl = Conformer(**mdl_config)
                else:
                    raise ValueError('Introduce a valid model')

                mdl.load_state_dict(torch.load(os.path.join(mdl_folder, mdl_filename), map_location=torch.device(device)))
                mdl.to(device)

                # LOAD THE DATA
                test_set = CustomDataset(dataset, data_path, 'test', subj, window=window_len, hop=hop, filt=filt, filt_path=filt_path, 
                                        leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, unit_output=unit_output)
                test_loader = DataLoader(test_set, batch_size, shuffle=not unit_output, pin_memory=True)
                
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

                        if dec_acc:
                            stimb = data['stimb'].to(device, dtype=torch.float)
                            unat_acc = correlation(stimb, y_hat, batch_dim=unit_output)
                            # Decoding accuracy
                            if acc.item() > unat_acc.item():
                                att_corr += 1

                        corr.append(acc.item())
                        nd_corr.append(nd_acc.item())

                eval_results[subj] = corr
                nd_results[subj] = nd_corr
                dec_accuracy = (att_corr / len(test_loader)) * 100
                dec_results.append(dec_accuracy)
                eval_mean_results.append(mean(corr))
                
                print(f'Subject {subj} | corr_mean {mean(corr)} | decode_accuracy {dec_accuracy}')

            if wandb_upload:
                wandb.log({'window': eval_window, 'corr_subj_mean': np.mean(eval_mean_results), 'corr_subj_std': np.std(eval_mean_results), 'decAcc_subj_mean': np.mean(dec_results), 'decAcc_subj_std': np.std(dec_results)})

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

        if wandb_upload: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default='configs/replicate_results/config.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    
    args = parser.parse_args()

    wandb_upload = args.wandb
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload=wandb_upload)