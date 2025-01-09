import torch
from utils.functional import multiple_loss_opt, set_seeds, get_data_path, get_channels, get_subjects, get_filename, get_loss
from utils.datasets import CustomDataset
from utils.loss_functions import CustomLoss
from torch.utils.data import DataLoader
from models.dnn import FCNN, CNN
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import os
import json
import wandb
import sys

def main(config, wandb_upload, dataset, key, early_stop, lr_decay=0.5):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'spatial_audio'
    
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
        mdl_load_path = os.path.join(global_path, 'results', project, key, 'models')

        # Load config
        ds_config = run['dataset_params']
        train_params = run['train_params']

        # Config training
        train_params = run['train_params']
        batch_size = train_params['batch_size']
        max_epoch = train_params['max_epoch']
        lr = float(train_params['lr'])
        weight_decay = float(train_params['weight_decay'])
        scheduler_patience = train_params['scheduler_patience'] if early_stop else 10
        early_stopping_patience = train_params['early_stopping_patience'] if early_stop else max_epoch
        loss_mode = train_params['loss_mode'] if 'loss_mode' in train_params.keys() else 'mean'
        alpha = train_params['alpha_loss'] if 'alpha_loss' in train_params.keys() else 0
        
        # Config dataset
        ds_config = run['dataset_params']
        preproc_mode = ds_config['preproc_mode'] if 'preproc_mode' in ds_config.keys() else None
        data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
        window_len = ds_config['window_len']
        hop = ds_config['hop']
        leave_one_out = True if key == 'subj_independent' else False
        data_type = ds_config['data_type'] if 'data_type' in ds_config.keys() else 'mat'
        eeg_band = ds_config['eeg_band'] if 'eeg_band' in ds_config.keys() else None
        fixed = ds_config['fixed']
        rnd_trials = ds_config['rnd_trials']
        hrtf = ds_config['hrtf'] if 'hrtf' in ds_config.keys() else False
        norm_hrtf_diff = ds_config['norm_hrtf_diff'] if 'norm_hrtf_diff' in ds_config.keys() else False
        window_pred = ds_config['window_pred'] if 'window_pred' in ds_config.keys() else not ds_config['unit_output']
        dec_acc = True if dataset != 'skl' else False # skl dataset without unattended stim => dec-acc is not possible
        val_hop = ds_config['hop'] if window_pred else 1
        shuffle = ds_config['shuffle'] if 'shuffle' in ds_config.keys() else True

        # Saving paths
        mdl_save_path = os.path.join(global_path, 'results', project, key, 'finetune_models')
        metrics_save_path = os.path.join(global_path, 'results', project, key, 'finetune_metrics')

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
        
        # Population mode that generates a model for all samples
        # if key == 'population':
        #     selected_subj = [get_subjects(dataset)]
        # else:
        #     selected_subj = get_subjects(dataset)
        selected_subj = get_subjects(dataset)[:5]

        for subj in selected_subj:

            print(f'Finetunning {model} on {subj} with {dataset} data...')
            
            mdl_folder = os.path.join(mdl_load_path, dataset_name+'_data', mdl_name)
            if key == 'population':
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
            mdl.finetune()
            mdl_size = sum(p.numel() for p in mdl.parameters())
            run['mdl_size'] = mdl_size
            run['subj'] = subj

            if wandb_upload: wandb.init(project=project, name=exp_name, tags=['finetune'], config=run)

            # LOAD THE DATA
            train_set = CustomDataset(dataset, data_path, 'train', subj, window=window_len, hop=hop, data_type=data_type, leave_one_out=leave_one_out,  
                                    fixed=fixed, rnd_trials = rnd_trials, window_pred=window_pred, hrtf=hrtf, eeg_band=eeg_band)
            val_set = CustomDataset(dataset, data_path, 'val',  subj, window=window_len, hop=val_hop, data_type=data_type, leave_one_out=leave_one_out, 
                                    fixed=fixed, rnd_trials = rnd_trials, window_pred=window_pred, hrtf=hrtf, eeg_band = eeg_band)
            
            train_loader = DataLoader(train_set, batch_size, shuffle= shuffle, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size, shuffle= window_pred, pin_memory=True)
            
            # OPTIMIZER PARAMS: optimize only parameters which contains grad
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr= lr * lr_decay, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

            # LOSS FUNCTION
            criterion = CustomLoss(mode=loss_mode, window_pred=window_pred, alpha_end=alpha)

            # Early stopping parameters
            best_accuracy=0
            best_epoch=0

            train_mean_loss = []
            val_mean_loss = []
            val_decAccuracies = []

            # FINETUNE THE MODEL
            for epoch in range(max_epoch):
                
                # Stop after n epoch without imporving the val loss
                if epoch > best_epoch + early_stopping_patience:
                    break

                mdl.train()
                train_loss = []

                # Initialize tqdm progress bar
                train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epoch}', leave=False, mininterval=0.5)

                for batch, data in enumerate(train_loader_tqdm):
                    
                    eeg = data['eeg'].to(device, dtype=torch.float)
                    stima = data['stima'].to(device, dtype=torch.float)

                    # Forward the model and calculate the loss corresponding to the neg. Pearson coef
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        preds = mdl(eeg)

                    loss_list = criterion(preds=preds, targets=stima)
                    loss = loss_list[0]

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Append neg. loss corresponding to the coef. Pearson
                    train_loss.append(loss_list)

                    # Actualize the state of the train loss
                    train_loader_tqdm.set_postfix({'train_loss': loss})

                mdl.eval()
                val_loss = []
                val_att_corr = 0

                # Validation
                with torch.no_grad():

                    for batch, data in enumerate(val_loader):

                        eeg = data['eeg'].to(device, dtype=torch.float)
                        stima = data['stima'].to(device, dtype=torch.float)

                        preds = mdl(eeg)

                        loss_list = criterion(preds=preds, targets=stima)
                        loss = loss_list[0]
                        
                        if dec_acc:
                            stimb = data['stimb'].to(device, dtype=torch.float)
                            # Decoding accuracy of the model
                            unat_loss = criterion(preds=preds, targets = stimb)[0]
                            if loss.item() < unat_loss.item():
                                val_att_corr += 1

                        val_loss.append(loss_list)

                mean_val_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in val_loss])).item()
                mean_train_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in train_loss])).item()
                val_decAccuracy = val_att_corr / len(val_loader) * 100

                scheduler.step(-mean_val_loss)

                # Logging metrics
                print(f'Epoch: {epoch} | Train loss: {-mean_train_loss:.4f} | Val loss/acc: {-mean_val_loss:.4f}/{val_decAccuracy:.4f}')
                
                if wandb_upload:
                    wandb_log = {'train_loss': mean_train_loss, 'val_loss': mean_val_loss, 'val_acc': val_decAccuracy}
                    # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
                    if multiple_loss_opt(loss_mode):
                        wandb_log['val_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in val_loss])).item()
                        wandb_log['train_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in train_loss])).item()
                        wandb_log['val_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in val_loss])).item()
                        wandb_log['train_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in train_loss])).item()
                    wandb.log(wandb_log)

                train_mean_loss.append(mean_train_loss)
                val_mean_loss.append(mean_val_loss)
                val_decAccuracies.append(val_decAccuracy)

                # Save best results
                if mean_val_loss < best_accuracy or epoch == 0:
                    # best_train_loss = mean_train_accuracy
                    best_accuracy = mean_val_loss
                    best_epoch = epoch
                    best_state_dict = mdl.state_dict()
            
            dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

            # Save best final model
            mdl_save_name = f'{model}_batch={batch_size}_block={window_len}_lr={lr}'
            # prefix = model if key == 'population' else subj
            prefix = subj

            # Add extensions to the model name depending on the params
            if preproc_mode is not None: mdl_save_name += '_' + preproc_mode
            if eeg_band is not None: mdl_save_name += '_' + eeg_band
            if loss_mode != 'mean': mdl_save_name += '_' + loss_mode
            if alpha != 0: mdl_save_name += '_alpha=' + str(alpha)
            if rnd_trials: mdl_save_name += '_rnd'
            if hrtf: mdl_save_name += '_hrtf'
        
            mdl_folder = os.path.join(mdl_save_path, dataset_filename+'_data', mdl_save_name)
            if not os.path.exists(mdl_folder):
                os.makedirs(mdl_folder)
            torch.save(
                best_state_dict, 
                os.path.join(mdl_folder, f'{prefix}_epoch={epoch}_acc={best_accuracy:.4f}.ckpt')
            )

            # Save corresponding train and val metrics
            val_folder = os.path.join(metrics_save_path, dataset_filename+'_data', mdl_save_name, 'val')
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)
            train_folder = os.path.join(metrics_save_path, dataset_filename+'_data', mdl_save_name, 'train')
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            json.dump(train_mean_loss, open(os.path.join(train_folder, f'{prefix}_train_loss_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))
            json.dump(val_mean_loss, open(os.path.join(val_folder, f'{prefix}_val_loss_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))
            json.dump(val_decAccuracies, open(os.path.join(val_folder, f'{prefix}_val_decAcc_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))

            if wandb_upload: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default='configs/spatial_audio/ild_tunning.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_specific', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--lr_decay", type=float, default=10, help="Scaling factor for the lr finetunning")
    parser.add_argument("--max_epoch", action='store_true', help="When included training performed for all the epoch without stop")
    
    args = parser.parse_args()

    wandb_upload = args.wandb
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    assert args.lr_decay < 100 and args.lr_decay > 0, 'lr decay must be a float numbet between 0 and 1'

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload, args.dataset, args.key, not args.max_epoch, args.lr_decay)