import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.dnn import FCNN
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from utils.functional import get_data_path, get_channels, get_subjects, correlation
from utils.datasets import CustomDataset
from tqdm import tqdm
import argparse
import yaml
import os
import json
import wandb

def main(config):

    for exp in config['experiments']:

        # Global params
        exp_name = exp['name']
        key = exp['key']
        global_path = exp['global_path']
        global_data_path = exp['global_data_path']
        dataset = exp['dataset']
        model = exp['model']

        # Config training
        train_params = exp['train_params']
        batch_size = train_params['batch_size']
        max_epoch = train_params['max_epoch']
        lr = float(train_params['lr'])
        weight_decay = float(train_params['weight_decay'])
        scheduler_patience = train_params['scheduler_patience']
        early_stopping_patience = train_params['early_stopping_patience']

        # Config dataset
        ds_config = exp['dataset_params']
        data_path = get_data_path(global_data_path, dataset, filt=False)
        window_len = ds_config['window_len']
        hop = ds_config['hop']
        leave_one_out = True if exp['key'] == 'subj_independent' else False
        filt = ds_config['filt']
        filt_path = get_data_path(global_data_path, dataset, filt=True)
        fixed = ds_config['fixed']
        rnd_trials = ds_config['rnd_trials']
        unit_output = ds_config['unit_output']

        # Saving paths
        mdl_save_path = global_path + '/results/'+exp['key']+'/models'
        metrics_save_path = global_path + '/results/'+exp['key']+'/metrics'

        # Population mode that generates a model for all samples
        if key == 'population':
            selected_subj = [get_subjects(dataset)]
        else:
            selected_subj = get_subjects(dataset)

        for subj in selected_subj:
            
            if key == 'population':
                print(f'Training {model} on all subjects with {dataset} data...')
            else:
                exp['subj'] = subj
                if key == 'subj_independent':
                    print(f'Training {model} leaving out {subj} with {dataset} data...')
                else:
                    print(f'Training {model} on {subj} with {dataset} data...')

            run = wandb.init(project='replicate_model_results', config=exp)

            # LOAD THE DATA
            train_set = CustomDataset(dataset, data_path, 'train', subj, window=window_len, hop=hop, filt=filt, filt_path=filt_path, 
                                        leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, unit_output=unit_output)
            val_set = CustomDataset(dataset, data_path, 'val',  subj, window=window_len, hop=hop, filt=filt, filt_path=filt_path, 
                                    leave_one_out=leave_one_out, fixed=fixed, rnd_trials = rnd_trials, unit_output=unit_output)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # LOAD THE MODEL
            match model:
                case 'FCNN':
                    exp['model_params']['n_chan'] = get_channels(dataset)
                    mdl = FCNN(**exp['model_params'])
                case 'VLAAI':
                    exp['model_params']['input_channels'] = get_channels(dataset)
                    mdl = VLAAI(**exp['model_params'])
                case 'EEG_Conformer':
                    exp['model_params']['eeg_channels'] = get_channels(dataset)
                    mdl_config = ConformerConfig(**exp['model_params'])
                    mdl = Conformer(**mdl_config)
                case _:
                    raise ValueError('Introduce a valid model')
            
            mdl.to(device)
            
            train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size, shuffle= not unit_output, pin_memory=True)

            optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

            # Early stopping parameters
            best_accuracy=0
            best_epoch=0

            train_mean_loss = []
            val_mean_loss = []
            train_decAccuracies = []
            val_decAccuracies = []

            # Training loop
            for epoch in range(max_epoch):
                
                # Stop after n epoch without imporving the val loss
                if epoch > best_epoch + early_stopping_patience:
                    break

                mdl.train()
                train_loss = []
                train_att_corr = 0

                # Initialize tqdm progress bar
                train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epoch}', leave=False, mininterval=0.5)

                for batch, data in enumerate(train_loader_tqdm):
                    
                    eeg = data['eeg'].to(device, dtype=torch.float)
                    stima = data['stima'].to(device, dtype=torch.float)
                    stimb = data['stimb'].to(device, dtype=torch.float)

                    # Forward the model and calculate the loss corresponding to the neg. Pearson coef
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        preds, loss = mdl(eeg, targets = stima)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Decoding accuracy of the model
                    unat_loss = correlation(stimb, preds, batch_dim=unit_output)
                    if loss.item() > unat_loss.item():
                        train_att_corr += 1
                    
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
                        stimb = data['stimb'].to(device, dtype=torch.float)

                        preds, loss = mdl(eeg, targets=stima)
                        
                        # Decoding accuracy of the model
                        unat_loss = correlation(stimb, preds, batch_dim=unit_output)
                        if loss.item() > unat_loss.item():
                            val_att_corr += 1

                        val_loss.append(-loss)

                mean_val_loss = torch.mean(torch.hstack(val_loss)).item()
                mean_train_loss = torch.mean(torch.hstack(train_loss)).item()
                train_decAccuracy = val_att_corr / len(val_loader)
                val_decAccuracy = train_att_corr / len(train_loader)

                scheduler.step(mean_val_loss)

                # Logging metrics
                print(f'Epoch: {epoch} | Train loss/acc: {mean_train_loss:.4f}/{train_decAccuracy:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f}')
                wandb.log({'epoch': epoch, 'train_loss': mean_train_loss, 'train_acc': train_decAccuracy, 'val_loss': mean_val_loss, 'val_acc': val_decAccuracy})
                
                train_mean_loss.append(mean_train_loss)
                val_mean_loss.append(mean_val_loss)
                train_decAccuracies.append(train_decAccuracy)
                val_decAccuracies.append(val_decAccuracy)

                # Save best results
                if val_mean_loss > best_accuracy or epoch == 0:
                    # best_train_loss = mean_train_accuracy
                    best_accuracy = val_mean_loss
                    best_epoch = epoch
                    best_state_dict = mdl.state_dict()
            
            dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

            # Save best final model
            mdl_name = f'{model}_batch={batch_size}_block={window_len}_lr={lr}'
            prefix = model if key == 'population' else subj

            # Add extensions to the model name depending on the params
            if filt:
                mdl_name = mdl_name + '_filt'
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
            json.dump(train_mean_loss, open(os.path.join(train_folder, f'{prefix}_train_decAcc_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))
            json.dump(val_mean_loss, open(os.path.join(val_folder, f'{prefix}_val_decAcc_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default='configs/replicate_results/config.yaml', help="Ruta al archivo config")
    
    args = parser.parse_args()
    
    # Upload results to wandb
    wandb.login()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config)