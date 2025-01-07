
import numpy as np
from tqdm import tqdm
from utils.loss_functions import CustomLoss
import argparse
import yaml
import wandb

from models.dnn import FCNN, CNN
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from utils.functional import multiple_loss_opt, get_data_path, get_channels, get_subjects, set_seeds, get_loss
from utils.datasets import CustomDataset

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def process_training_run(run, config, dataset, global_data_path, project, key, early_stop):

    # Global params
    model = run['model']
    exp_name = config['exp_name'] + '_' + str(run['loss_params']['alpha_end'])

    wandb.init(config=config)

    # Config training
    train_params = run['train_params']
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    max_epoch = train_params['max_epoch']
    weight_decay = float(train_params['weight_decay'])
    scheduler_patience = train_params['scheduler_patience'] if early_stop else 10
    early_stopping_patience = train_params['early_stopping_patience'] if early_stop else max_epoch
    preproc_mode = train_params.get('preproc_mode')

    # Config dataset
    ds_config = run['dataset_params']
    run['dataset_params']['window'] = wandb.config.window
    ds_config['leave_one_out'] = True if key == 'subj_independent' else False

    # Config model
    run['model_params']['dropout'] = wandb.config.dropout
    run['model_params']['input_samples'] = wandb.config.window
    run['model_params']['F1'] = wandb.config.F1
    run['model_params']['F2'] = wandb.config.F2
    run['model_params']['AP1'] = wandb.config.AP1

    # Config loss
    loss_params = run['loss_params']
    run['loss_params']['alpha_end'] = wandb.config.alpha_loss
    loss_mode = loss_params.get('mode', 'mean')
    
    data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)

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
            mdl = FCNN(**run['model_params'])
        elif model == 'CNN':
            run['model_params']['input_channels'] = get_channels(dataset)
            mdl = CNN(**run['model_params'])
        elif model == 'VLAAI':
            run['model_params']['input_channels'] = get_channels(dataset)
            mdl = VLAAI(**run['model_params'])
        elif model == 'Conformer':
            run['model_params']['eeg_channels'] = get_channels(dataset)
            run['model_params']['kernel_chan'] = get_channels(dataset)
            mdl_config = ConformerConfig(**run['model_params'])
            mdl = Conformer(mdl_config)
        else:
            raise ValueError('Introduce a valid model')

        mdl.to(device)
        mdl_size = sum(p.numel() for p in mdl.parameters())
        run['mdl_size'] = mdl_size
        run['subject'] = subj
        print(f'Model size: {mdl_size / 1e06:.2f}M')

        # WEIGHT INITIALIZATION
        if exp_name == 'bool_params':
            if train_params['init_weights']: mdl.apply(mdl.init_weights)

        # LOAD THE DATA
        train_set = CustomDataset(
            dataset=dataset,
            data_path=data_path,
            split='train',
            subjects=subj,
            **run['dataset_params']
        )
        val_set = CustomDataset(
            dataset=dataset,
            data_path=data_path,
            split='val',
            subjects=subj,
            **run['dataset_params']
        )

        train_loader = DataLoader(train_set, batch_size, shuffle=train_params.get('shuffle', True), pin_memory=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=ds_config.get('window_pred', True), pin_memory=True)

        # OPTIMIZER PARAMS
        optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

        # LOSS FUNCTION
        criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

        # Early stopping parameters
        best_epoch = 0

        # Training loop
        for epoch in range(max_epoch):

            # Stop after n epoch without improving the val loss
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

                    if ds_config.get('dec_acc', True):
                        stimb = data['stimb'].to(device, dtype=torch.float)
                        # Decoding accuracy of the model
                        unat_loss = criterion(preds=preds, targets=stimb)[0]
                        if loss.item() < unat_loss.item():
                            val_att_corr += 1

                    val_loss.append(loss_list)

            mean_val_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in val_loss])).item()
            mean_train_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in train_loss])).item()
            val_decAccuracy = val_att_corr / len(val_loader) * 100

            scheduler.step(-mean_val_loss)

            # Logging metrics
            print(f'Epoch: {epoch} | Train loss: {-mean_train_loss:.4f} | Val loss/acc: {-mean_val_loss:.4f}/{val_decAccuracy:.4f}')

            wandb_log = {'epoch': epoch,'train_loss': mean_train_loss, 'val_loss': mean_val_loss, 'val_acc': val_decAccuracy}
            # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
            if multiple_loss_opt(loss_mode):
                wandb_log['val_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in val_loss])).item()
                wandb_log['train_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in train_loss])).item()
                wandb_log['val_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in val_loss])).item()
                wandb_log['train_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in train_loss])).item()
            wandb.log(wandb_log)

        wandb.finish()

def main(config, dataset, key):

    global_data_path = config['global_data_path']
    project = 'spatial_audio'
    config['dataset'] = dataset
    config['key'] = key

    # REPRODUCIBILITY
    if 'seed' in config.keys(): 
        set_seeds(config['seed'])
        exp_name =  exp_name + '_' + config['seed']
    else: 
        set_seeds() # default seed = 42

    for run in config['runs']:
        process_training_run(run, config, dataset, global_data_path, project, key, early_stop=True)

if __name__ == "__main__":

    config_path = 'configs/spatial_audio/sweep_config.yaml'
    dataset = 'fulsang'
    key = 'population'

    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Upload results to wandb
    wandb.login() 

    # Load corresponding config
    with open(config_path, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    # Ejecutar agentes
    main(config, dataset, key)
