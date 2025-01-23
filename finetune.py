import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.functional import load_model, get_mdl_name, multiple_loss_opt, set_seeds, get_data_path, get_channels, get_subjects, get_filename, get_loss
from utils.datasets import CustomDataset
from utils.loss_functions import CustomLoss
from utils.sampler import BatchRandomSampler

from tqdm import tqdm
import argparse
import yaml
import os
import json
import wandb


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

        # Get configs
        train_config = run['train_params']
        ds_config = run['dataset_params']
        loss_config = run['loss_params']

        # Config training
        batch_size = train_config['batch_size']
        lr = float(train_config['lr'])
        max_epoch = train_config.get('max_epoch', 200)
        weight_decay = float(train_config.get('weight_decay', 1e-8))
        scheduler_patience = train_config.get('scheduler_patience', 10)
        early_stopping_patience = train_config['early_stopping_patience'] if early_stop else max_epoch
        batch_rnd_sampler = train_config.get('batch_rnd_sampler', False)
        preproc_mode = train_config.get('preproc_mode')
        shuffle = train_config.get('shuffle', False)
        val_shuffle = shuffle if ds_config.get('window_pred') else False

        # Config dataset
        ds_config['leave_one_out'] = True if key == 'subj_independent' else False
        ds_val_config = ds_config
        if ds_config['window_pred'] == False: ds_val_config['hop'] = 1 

        # Config loss
        loss_mode = loss_config.get('mode', 'mean')

        # Saving paths
        mdl_save_path = os.path.join(global_path, 'results', project, key, 'finetune_models')
        metrics_save_path = os.path.join(global_path, 'results', project, key, 'finetune_metrics')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # GET THE MODEL PATH
        mdl_name = get_mdl_name(run)

        # Extract the finetunning loss if it's different after getting the model's name
        run['loss_params']['mode'] = run.get('finetune_loss', loss_mode) # for the mdl_name when saving
        loss_mode = run.get('finetune_loss', loss_mode)

        # Finetune the model with each subjetc of the dataset
        selected_subj = get_subjects(dataset)

        for subj in selected_subj:

            print(f'Finetunning {model} on {subj} with {dataset} data...')
            
            mdl_folder = os.path.join(mdl_load_path, dataset+'_data', mdl_name)
            
            if key == 'population':
                mdl_filename = os.listdir(mdl_folder)[0] # only a single trained model
            else:
                mdl_filename = get_filename(mdl_folder, subj) # search for the model related with the subj
            
            # LOAD THE MODEL
            mdl = load_model(run, dataset)

            mdl.load_state_dict(torch.load(os.path.join(mdl_folder, mdl_filename), map_location=torch.device(device)))
            mdl.to(device)
            mdl.finetune()
            mdl_size = sum(p.numel() for p in mdl.parameters())
            run['mdl_size'] = mdl_size
            run['subj'] = subj

            if wandb_upload: wandb.init(project=project, name=exp_name, tags=['finetune'], config=run)

            # LOAD THE DATA
            data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
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
            
            val_loader = DataLoader(val_set, batch_size, shuffle = val_shuffle, pin_memory=True, drop_last=True)
            
            # OPTIMIZER PARAMS: optimize only parameters which contains grad
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr= lr * lr_decay, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

            # LOSS FUNCTION
            criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

            # Early stopping parameters
            best_accuracy=0
            best_epoch=0

            # Saving metrics lists
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
                    stimb = data['stimb'].to(device, dtype=torch.float)

                    # Forward the model and calculate the loss corresponding to the neg. Pearson coef
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        preds = mdl(eeg)
                    
                    targets = [stima, stimb] if 'penalty' in loss_mode else stima

                    loss_list = criterion(preds=preds, targets=targets)
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
                val_att_loss = 0
                val_att_ild = 0
                val_att_corr = 0

                # Validation
                with torch.no_grad():

                    for batch, data in enumerate(val_loader):

                        eeg = data['eeg'].to(device, dtype=torch.float)
                        stima = data['stima'].to(device, dtype=torch.float)
                        stimb = data['stimb'].to(device, dtype=torch.float)

                        preds = mdl(eeg)

                        targets = [stima, stimb] if 'penalty' in loss_mode else stima

                        loss_list = criterion(preds=preds, targets = targets)
                        loss = loss_list[0]
                        
                        if stimb is not None:
                            if loss_mode != 'spatial_locus':
                                # Decoding accuracy of the model
                                targets_unatt = [stimb, stima] if 'penalty' in loss_mode else stimb
                                unat_loss_list = criterion(preds=preds, targets = targets_unatt)
                                unat_loss = unat_loss_list[0]
                                if loss.item() < unat_loss.item():
                                    val_att_loss += 1
                                # When multiple loss compute the accuracy for each metric
                                if len(loss_list) > 1:
                                    if loss_list[1].item() < unat_loss_list[1].item():
                                        val_att_corr += 1
                                    if loss_list[2].item() < unat_loss_list[2].item():
                                        val_att_ild += 1
                            else:
                                probs = F.sigmoid(preds)
                                pred_labels = (probs >= 0.5).float()
                                n_correct = torch.sum(pred_labels == stima)
                                if n_correct > batch_size // 2:
                                    val_att_loss += 1

                        val_loss.append(loss_list)

                mean_val_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in val_loss])).item()
                mean_train_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in train_loss])).item()
                val_decAccuracy = val_att_loss / len(val_loader) * 100
                val_corr_decAccuracy = val_att_corr / len(val_loader) * 100
                val_ild_decAccuracy = val_att_ild / len(val_loader) * 100

                scheduler.step(-mean_val_loss)

                # Logging metrics
                if multiple_loss_opt(loss_mode):
                    print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f} | Val ild_acc: {val_ild_decAccuracy} | Val corr_acc: {val_corr_decAccuracy}')
                else:
                    print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f}')
                
                if wandb_upload:
                    wandb_log = {'train_loss': mean_train_loss, 'val_loss': mean_val_loss, 'val_acc': val_decAccuracy}
                    # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
                    if multiple_loss_opt(loss_mode):
                        wandb_log['val_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in val_loss])).item()
                        wandb_log['train_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in train_loss])).item()
                        wandb_log['val_acc_corr'] = val_corr_decAccuracy
                        wandb_log['val_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in val_loss])).item()
                        wandb_log['train_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in train_loss])).item()
                        wandb_log['val_acc_ild'] = val_ild_decAccuracy
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
            
            # Save best final model
            mdl_save_name = get_mdl_name(run)
            prefix = subj
        
            mdl_folder = os.path.join(mdl_save_path, dataset+'_data', mdl_save_name)
            if not os.path.exists(mdl_folder):
                os.makedirs(mdl_folder)
            torch.save(
                best_state_dict, 
                os.path.join(mdl_folder, f'{prefix}_epoch={epoch}_acc={best_accuracy:.4f}.ckpt')
            )

            # Save corresponding train and val metrics
            val_folder = os.path.join(metrics_save_path, dataset+'_data', mdl_save_name, 'val')
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)
            train_folder = os.path.join(metrics_save_path, dataset+'_data', mdl_save_name, 'train')
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
    parser.add_argument("--config", type=str, default='configs/finetunning/ild_penalty.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
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