
import numpy as np
from tqdm import tqdm
from utils.loss_functions import CustomLoss
import argparse
import yaml
import wandb

from utils.functional import load_model, multiple_loss_opt, get_data_path, get_channels, get_subjects, set_seeds, get_loss
from utils.datasets import CustomDataset
from utils.sampler import BatchRandomSampler

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def process_training_run(run, config, dataset, global_data_path, project, key, early_stop):

    # Global params
    model = run['model']
    exp_name = config['exp_name']

    wandb.init(config=config)

    # Config training
    train_params = run['train_params']
    lr = float(getattr(wandb.config, 'lr', train_params.get('lr')))
    batch_size = getattr(wandb.config, 'batch_size', train_params.get('batch_size'))
    max_epoch = train_params['max_epoch']
    weight_decay = float(train_params['weight_decay'])
    scheduler_patience = train_params.get('scheduler_patience', 10)
    early_stopping_patience = train_params['early_stopping_patience'] if early_stop else max_epoch
    preproc_mode = train_params.get('preproc_mode')
    batch_rnd_sampler = train_params.get('batch_rnd_sampler', False)
    shuffle = train_params.get('shuffle', False)

    # Config dataset
    ds_config = run['dataset_params']
    run['dataset_params']['window'] = getattr(wandb.config, 'window', ds_config['window'])
    ds_config['leave_one_out'] = True if key == 'subj_independent' else False
    ds_val_config = ds_config
    if ds_config['window_pred'] == False: ds_val_config['hop'] = 1 
    val_shuffle = shuffle if ds_config.get('window_pred') else False

    # Config loss
    loss_params = run['loss_params']
    loss_params['alpha_end'] = getattr(wandb.config, 'alpha_end', loss_params.get('alpha_end'))
    loss_params['mode'] = getattr(wandb.config, 'mode', loss_params.get('mode', 'mean'))
    
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
        mdl = load_model(run, dataset)

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
            **ds_config
        )
        val_set = CustomDataset(
            dataset=dataset,
            data_path=data_path,
            split='val',
            subjects=subj,
            **ds_val_config
        )

        if batch_rnd_sampler:
            batch_sampler = BatchRandomSampler(train_set, batch_size)
            train_loader = DataLoader(train_set, batch_sampler=batch_sampler, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size, shuffle = shuffle, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size, shuffle= val_shuffle, pin_memory=True)

        # OPTIMIZER PARAMS
        optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

        # LOSS FUNCTION
        criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

        # Early stopping parameters
        best_epoch = 0
        best_accuracy = 0

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

                # Update the state of the train loss
                train_loader_tqdm.set_postfix({'train_loss': loss.item()})

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
            if multiple_loss_opt(loss_params['mode']):
                wandb_log['val_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in val_loss])).item()
                wandb_log['train_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in train_loss])).item()
                wandb_log['val_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in val_loss])).item()
                wandb_log['train_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in train_loss])).item()
            wandb.log(wandb_log)

            # Save best results
            if mean_val_loss < best_accuracy or epoch == 0:
                best_accuracy = mean_val_loss
                best_epoch = epoch

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

    config_path = 'configs/sweep_runs/ild_sweep_local.yaml'
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
