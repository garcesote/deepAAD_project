import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.dnn import FCNN, CNN
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from utils.functional import get_data_path, get_channels, get_subjects, set_seeds, correlation
from utils.datasets import CustomDataset
from tqdm import tqdm
import argparse
import yaml
import os
import json
import wandb

def main(config, wandb_upload, dataset, key, tunning, gradient_tracking, early_stop):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'gradient_tracking'
    exp_name = config['exp_name'] + '_' + dataset
    config['dataset'] = dataset
    config['key'] = key

    # REPRODUCIBILITY
    if 'seed' in config.keys(): 
        set_seeds(config['seed'])
        exp_name =  exp_name + '_' + config['seed']
    else: 
        set_seeds() # default seed = 42

    for run in config['runs']:

        # Global params
        model = run['model']
        # exp_name = ('_').join([key, dataset, model])

        # Config training
        train_params = run['train_params']
        batch_size = train_params['batch_size']
        max_epoch = train_params['max_epoch']
        lr = float(train_params['lr'])
        weight_decay = float(train_params['weight_decay'])
        scheduler_patience = train_params['scheduler_patience'] if early_stop else 10
        early_stopping_patience = train_params['early_stopping_patience'] if early_stop else max_epoch

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
        unit_output = ds_config['unit_output']
        dec_acc = True if dataset != 'skl' else False # skl dataset without unattended stim => dec-acc is not possible
        val_hop = ds_config['hop'] if not unit_output else 1
        # val_bs = 1 if not unit_output and dec_acc else batch_size # batch size equals 1 when estamiting a window instead of a single output

        # Saving paths
        mdl_save_path = os.path.join(global_path, 'results', exp_name, key, 'models')
        metrics_save_path = os.path.join(global_path, 'results', exp_name, key, 'metrics')

        # Population mode that generates a model for all samples
        if key == 'population':
            selected_subj = [get_subjects(dataset)]
        else:
            selected_subj = get_subjects(dataset)

        for subj in selected_subj:
            
            if key == 'population':
                print(f'Training {model} on all subjects with {dataset} data...')
            else:
                run['subj'] = subj
                if key == 'subj_independent':
                    print(f'Training {model} leaving out {subj} with {dataset} data...')
                else:
                    print(f'Training {model} on {subj} with {dataset} data...')
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # LOAD THE MODEL
            if model == 'FCNN':
                run['model_params']['n_chan'] = get_channels(dataset)
                run['model_params']['unit_output'] = unit_output
                mdl = FCNN(**run['model_params'])
            elif model == 'CNN':
                run['model_params']['input_channels'] = get_channels(dataset)
                run['model_params']['unit_output'] = unit_output
                mdl = CNN(**run['model_params'])
            elif model == 'VLAAI':
                run['model_params']['input_channels'] = get_channels(dataset)
                mdl = VLAAI(**run['model_params'])
            elif model == 'Conformer':
                run['model_params']['eeg_channels'] = get_channels(dataset)
                run['model_params']['kernel_chan'] = get_channels(dataset)
                run['model_params']['unit_output'] = unit_output
                mdl_config = ConformerConfig(**run['model_params'])
                mdl = Conformer(mdl_config)
            else:
                raise ValueError('Introduce a valid model')
            
            mdl.to(device)
            mdl_size = sum(p.numel() for p in mdl.parameters())
            run['mdl_size'] = mdl_size
            print(f'Model size: {mdl_size / 1e06:.2f}M')
            # WEIGHT INITILAIZATION
            if exp_name == 'bool_params':
                if train_params['init_weights']: mdl.apply(mdl.init_weights)
    
            if wandb_upload: wandb.init(project=project, name=exp_name, tags=['training'], config=run)
            if gradient_tracking and wandb_upload: wandb.watch(models=mdl, log='all')

            # LOAD THE DATA
            train_set = CustomDataset(dataset, data_path, 'train', subj, window=window_len, hop=hop, data_type=data_type, leave_one_out=leave_one_out,  
                                       fixed=fixed, rnd_trials = rnd_trials, unit_output=unit_output, eeg_band=eeg_band)
            val_set = CustomDataset(dataset, data_path, 'val',  subj, window=window_len, hop=val_hop, data_type=data_type, leave_one_out=leave_one_out, 
                                    fixed=fixed, rnd_trials = rnd_trials, unit_output=unit_output, eeg_band = eeg_band)
            
            train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size, shuffle= not unit_output, pin_memory=True)

            optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)
            # optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

            # Early stopping parameters
            best_accuracy=0
            best_epoch=0

            train_mean_loss = []
            val_mean_loss = []
            val_decAccuracies = []

            # Training loop
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
                        preds, loss = mdl(eeg, targets = stima)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Append neg. loss corresponding to the coef. Pearson
                    train_loss.append(-loss)

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

                        preds, loss = mdl(eeg, targets=stima)
                        
                        if dec_acc:
                            stimb = data['stimb'].to(device, dtype=torch.float)
                            # Decoding accuracy of the model
                            unat_loss = correlation(stimb, preds, batch_dim=unit_output)
                            if -loss.item() > unat_loss.item():
                                val_att_corr += 1

                        val_loss.append(-loss)

                mean_val_loss = torch.mean(torch.hstack(val_loss)).item()
                mean_train_loss = torch.mean(torch.hstack(train_loss)).item()
                val_decAccuracy = val_att_corr / len(val_loader) * 100

                scheduler.step(mean_val_loss)

                # Logging metrics
                print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f}')
                
                if wandb_upload:
                    wandb.log({'train_loss': -mean_train_loss, 'val_loss': -mean_val_loss, 'val_acc': val_decAccuracy})
            
                train_mean_loss.append(mean_train_loss)
                val_mean_loss.append(mean_val_loss)
                val_decAccuracies.append(val_decAccuracy)

                # Save best results
                if mean_val_loss > best_accuracy or epoch == 0:
                    # best_train_loss = mean_train_accuracy
                    best_accuracy = mean_val_loss
                    best_epoch = epoch
                    best_state_dict = mdl.state_dict()
            
            dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

            if not tunning:
                # Save best final model
                mdl_name = f'{model}_batch={batch_size}_block={window_len}_lr={lr}'
                prefix = model if key == 'population' else subj

                # Add extensions to the model name depending on the params
                if 'preproc_mode' in ds_config.keys():
                    mdl_name = mdl_name + '_' + preproc_mode
                if eeg_band is not None:
                    mdl_name = mdl_name + '_' + eeg_band
                if rnd_trials:
                    mdl_name = mdl_name + '_rnd'
            
                mdl_folder = os.path.join(mdl_save_path, dataset_filename+'_data', mdl_name)
                if not os.path.exists(mdl_folder):
                    os.makedirs(mdl_folder)
                torch.save(
                    best_state_dict, 
                    os.path.join(mdl_folder, f'{prefix}_epoch={epoch}_acc={best_accuracy:.4f}.ckpt')
                )

                # Save corresponding train and val metrics
                val_folder = os.path.join(metrics_save_path, dataset_filename+'_data', mdl_name, 'val')
                if not os.path.exists(val_folder):
                    os.makedirs(val_folder)
                train_folder = os.path.join(metrics_save_path, dataset_filename+'_data', mdl_name, 'train')
                if not os.path.exists(train_folder):
                    os.makedirs(train_folder)
                json.dump(train_mean_loss, open(os.path.join(train_folder, f'{prefix}_train_loss_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))
                json.dump(val_mean_loss, open(os.path.join(val_folder, f'{prefix}_val_loss_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))
                json.dump(val_decAccuracies, open(os.path.join(val_folder, f'{prefix}_val_decAcc_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))

            if wandb_upload: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default="configs/replicate_results/train_vlaai.yaml", help="Ruta al archivo config")
    # parser.add_argument("--config", type=str, default='configs/gradient_tracking/models_tracking.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--tunning", action='store_true', help="When included do not save results on local folder")
    parser.add_argument("--gradient_tracking", action='store_true', help="When included register gradien on wandb")
    parser.add_argument("--max_epoch", action='store_true', help="When included training performed for all the epoch without stop")
    parser.add_argument("--dataset", type=str, default='skl', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_specific', help="Key from subj_specific, subj_independent and population")
    
    args = parser.parse_args()

    wandb_upload = args.wandb
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload, args.dataset, args.key, args.tunning, args.gradient_tracking, not args.max_epoch)