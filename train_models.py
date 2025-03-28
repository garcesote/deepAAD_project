import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.functional import verbose, load_model, get_mdl_name, multiple_loss_opt, get_data_path, get_subjects, set_seeds
from utils.datasets import CustomDataset, CustomPathDataset
from utils.sampler import BatchRandomSampler
from utils.loss_functions import CustomLoss

from tqdm import tqdm
import argparse
import yaml
import os
import json
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(config, wandb_upload, dataset, key, cross_val, tunning, gradient_tracking, early_stop, project):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    exp_name = config['exp_name']

    # REPRODUCIBILITY
    if 'seed' in config.keys(): 
        set_seeds(config['seed'])
        exp_name =  exp_name + '_' + config['seed']
    else: 
        set_seeds() # default seed = 42

    for run in config['runs']:

        # Global params
        model = run['model']

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
        num_workers = train_config.get('num_workers', 0)
        val_step = train_config.get('val_step', 1)
        path_dataset = train_config.get('path_dataset', False)
        if path_dataset:
            val_shuffle = True
            val_batch_size = 1
        else:
            val_shuffle = shuffle if ds_config.get('window_pred') else False
            val_batch_size = 1 if ds_config.get('window_pred') else batch_size

        # Config dataset
        ds_config['leave_one_out'] = True if key == 'subj_independent' else False
        ds_val_config = ds_config.copy()
        if not ds_config.get('window_pred', True): ds_val_config['hop'] = 1
        stim_input = ds_config.get('stim_input', False) 

        # Config loss
        loss_mode = loss_config.get('mode', 'mean')
        # Exception on val_batch_size on window pred when using triplet loss
        if loss_mode == 'triplet_loss': val_batch_size = batch_size 
        
        # Saving paths
        mdl_save_path = os.path.join(global_path, 'results', project, key, 'models')
        metrics_save_path = os.path.join(global_path, 'results', project, key, 'metrics')

        # Population mode that generates a model for all samples
        if key == 'population':
            selected_subj = [get_subjects(dataset)]
        else:
            selected_subj = get_subjects(dataset)

        # Cross validation 
        if cross_val:
            n_folds = 5 if dataset == 'fulsang' else 4
        else:
            n_folds = 1

        for cv_fold in range(n_folds):
            
            if not cross_val: cv_fold = None

            for subj in selected_subj:
                
                # VERBOSE
                verbose('train', key, subj, dataset, model, loss_mode=loss_mode, cv_fold=cv_fold)

                # LOAD THE MODEL
                mdl = load_model(run, dataset, wandb_upload, sweep=False)
                mdl.to(device)
                mdl_size = sum(p.numel() for p in mdl.parameters())
                print(f'Model size: {mdl_size / 1e06:.2f}M')
                
                # WEIGHT INITILAIZATION
                init_weights = train_config.get('init_weights', False)
                if init_weights: mdl.init_weights()

                # WANDB INIT
                run['subject'] = subj
                if cross_val: run['cv_fold'] = cv_fold
                run['key'] = key
                run['dataset'] = dataset
                run['mdl_size'] = mdl_size
                if wandb_upload: wandb.init(project=project, name=exp_name, tags=['training'], config=run)

                if gradient_tracking and wandb_upload: wandb.watch(models=mdl, log='all')

                # LOAD THE DATA
                data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
                ds_args = dict(
                    dataset=dataset,
                    data_path=data_path,
                    subjects=subj,
                    **ds_config
                )
                if path_dataset:
                    train_set = CustomPathDataset(split='train', **ds_args)
                    val_set = CustomPathDataset(split='val', **ds_args)
                else:
                    train_set = CustomDataset(split='train', cv_fold=cv_fold, **ds_args)
                    val_set = CustomDataset(split='val', cv_fold=cv_fold, **ds_args)
                
                if batch_rnd_sampler:
                    batch_sampler = BatchRandomSampler(train_set, batch_size)
                    train_loader = DataLoader(train_set, batch_sampler=batch_sampler, pin_memory=True, num_workers=num_workers)
                else:
                    train_loader = DataLoader(train_set, batch_size, shuffle = shuffle, pin_memory=True, drop_last=True, num_workers=num_workers)
                
                val_loader = DataLoader(val_set, val_batch_size, shuffle = val_shuffle, pin_memory=True, drop_last=True, num_workers=num_workers)
                
                # OPTIMIZER PARAMS
                optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

                # LOSS FUNCTION
                criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

                # Early stopping parameters
                best_loss = 0
                best_accuracy=0
                best_epoch=0

                # Saving metrics lists
                train_mean_loss = []
                val_mean_loss = []
                val_decAccuracies = []
            
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
                        
                        # Load the data
                        eeg = data['eeg'].to(device, dtype=torch.float)
                        stima = data['stima'].to(device, dtype=torch.float)
                        if dataset != 'skl': stimb = data['stimb'].to(device, dtype=torch.float)

                        # Forward pass depending on the model inputs
                        if stim_input:
                            assert dataset != 'skl', "Not possible to forward the model with skl dataset as there's no unatt stim"
                            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                preds = mdl(eeg, stima.unsqueeze(1), stimb.unsqueeze(1))
                            targets = data['labels'].to(device, dtype=torch.float)
                        else:
                            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                preds = mdl(eeg)
                            targets = [stima, stimb] if 'penalty' in loss_mode else stima

                        # Compute the loss
                        loss_list = criterion(preds, targets)
                        loss = loss_list[0]

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Append neg. loss corresponding to the coef. Pearson
                        train_loss.append(loss_list)
                        # Update the state of the train loss 
                        train_loader_tqdm.set_postfix({'train_loss': loss.item()})
                    
                    # Compute the train loss mean
                    mean_train_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in train_loss])).item()

                    # Evaluate after val_step epochs for training
                    if epoch == 0 or epoch % val_step == 0:

                        mdl.eval()
                        val_loss = []
                        val_att_loss = 0
                        val_att_ild = 0
                        val_att_corr = 0

                        # Initialize tqdm progress bar
                        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch}/{max_epoch}', leave=False, mininterval=0.5)

                        # Validation
                        with torch.no_grad():

                            for batch, data in enumerate(val_loader_tqdm):

                                eeg = data['eeg'].to(device, dtype=torch.float)
                                stima = data['stima'].to(device, dtype=torch.float)
                                if dataset != 'skl': stimb = data['stimb'].to(device, dtype=torch.float)

                                # Get model predictions
                                if stim_input:
                                    assert dataset != 'skl', "Not possible to forward the model with skl dataset as there's no unatt stim"
                                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                        preds = mdl(eeg, stima.unsqueeze(1), stimb.unsqueeze(1))
                                    targets = data['labels'].to(device, dtype=torch.float)
                                else:
                                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                        preds = mdl(eeg)
                                    targets = [stima, stimb] if 'penalty' in loss_mode else stima
                                
                                # Compute the loss
                                loss_list = criterion(preds, targets)
                                loss = loss_list[0]

                                # Decode accuracy
                                if dataset != 'skl' and loss_mode != 'triplet_loss':
                                    # Direct classification based on the predictions
                                    if loss_mode == 'spatial_locus':
                                        probs = F.sigmoid(preds)
                                        pred_labels = (probs >= 0.5).float()
                                        n_correct = torch.sum(pred_labels == stima).float()
                                        if n_correct > batch_size / 2:
                                            val_att_loss += 1
                                    elif stim_input:
                                        pred_labels = (preds >= 0.5).float()
                                        n_correct = torch.sum(pred_labels == targets).float()
                                        if n_correct > val_batch_size / 2:
                                            val_att_loss += 1

                                    # Classification by comparing loss
                                    else:
                                        unat_loss_list = criterion(preds=preds, targets = stimb)
                                        unat_loss = unat_loss_list[0]
                                        if loss.item() < unat_loss.item():
                                            val_att_loss += 1
                                        # When multiple loss compute the accuracy for each metric
                                        if len(loss_list) > 1:
                                            if loss_list[1].item() < unat_loss_list[1].item():
                                                val_att_corr += 1
                                            if loss_list[2].item() < unat_loss_list[2].item():
                                                val_att_ild += 1

                                val_loss.append(loss_list)
                                val_loader_tqdm.set_postfix({'val_loss': loss.item()})

                        mean_val_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in val_loss])).item()
                        val_decAccuracy = val_att_loss / len(val_loader) * 100
                        val_corr_decAccuracy = val_att_corr / len(val_loader) * 100
                        val_ild_decAccuracy = val_att_ild / len(val_loader) * 100

                        scheduler.step(-mean_val_loss)

                        # Save best results
                        if mean_val_loss < best_loss or epoch == 0:
                            # best_train_loss = mean_train_accuracy
                            best_loss = mean_val_loss
                            best_accuracy = val_decAccuracy
                            best_epoch = epoch
                            best_state_dict = mdl.state_dict()

                    # Logging metrics
                    if multiple_loss_opt(loss_mode):
                        print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f} | Val ild_acc: {val_ild_decAccuracy} | Val corr_acc: {val_corr_decAccuracy}')
                    else:
                        print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f}')
                    
                    if wandb_upload:
                        wandb_log = {'train_loss': mean_train_loss}
                        # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
                        if epoch == 0 or epoch % val_step == 0:
                            wandb_log['val_loss'] = mean_val_loss
                            wandb_log['val_acc'] = val_decAccuracy
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

                # Save best final model and metrics           
                if not tunning:

                    mdl_name = get_mdl_name(run)
                    prefix = model if not cross_val else f'{model}_cvFold={cv_fold}'
                    mdl_folder = os.path.join(mdl_save_path, dataset+'_data', mdl_name)
                    if key != 'population': mdl_folder = os.path.join(mdl_folder, subj) # Add subj folder CNN[...]/S1/

                    if not os.path.exists(mdl_folder):
                        os.makedirs(mdl_folder)
                    torch.save(
                        best_state_dict, 
                        os.path.join(mdl_folder, f'{prefix}_epoch={best_epoch}_loss={best_loss:.4f}_acc={best_accuracy:.4f}.ckpt')
                    )

                    metrics_folder = os.path.join(metrics_save_path, dataset+'_data', mdl_name)
                    if key != 'population': metrics_folder = os.path.join(metrics_folder, subj) # Add subj folder CNN[...]/S1/

                    val_folder = os.path.join(metrics_folder, 'val')
                    if not os.path.exists(val_folder):
                        os.makedirs(val_folder)
                    train_folder = os.path.join(metrics_folder, 'train')
                    if not os.path.exists(train_folder):
                        os.makedirs(train_folder)
                    json.dump(train_mean_loss, open(os.path.join(train_folder, f'{prefix}_train_loss_epoch={best_epoch}_loss={best_loss:.4f}_acc={best_accuracy:.4f}.ckpt'),'w'))
                    json.dump(val_mean_loss, open(os.path.join(val_folder, f'{prefix}_val_loss_epoch={best_epoch}_loss={best_loss:.4f}_acc={best_accuracy:.4f}.ckpt'),'w'))
                    json.dump(val_decAccuracies, open(os.path.join(val_folder, f'{prefix}_val_acc_epoch={best_epoch}_loss={best_loss:.4f}_acc={best_accuracy:.4f}.ckpt'),'w'))

                if wandb_upload: 
                    wandb.log({'best_epoch': best_epoch, 'best_loss': best_loss, 'best_acc':best_accuracy})
                    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default="configs/euroacustics/cnn.yaml", help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--cross_val", action='store_true', help="When included perform a 5 cross validation for the train_set")
    parser.add_argument("--tunning", action='store_true', help="When included do not save results on local folder")
    parser.add_argument("--gradient_tracking", action='store_true', help="When included register gradien on wandb")
    parser.add_argument("--sync", action='store_true', help="When included register gradien on wandb")    
    parser.add_argument("--max_epoch", action='store_true', help="When included training performed for all the epoch without stop")
    parser.add_argument("--dataset", type=str, default='jaulab', help="Dataset")
    parser.add_argument("--key", type=str, default='subj_independent', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--project", type=str, default='stim_input', help="Name of the project that must corresponds with wandb project and set the save path for metrics and models")

    args = parser.parse_args()

    wandb_upload = args.wandb
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    if args.sync: torch.cuda.synchronize()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload, args.dataset, args.key, args.cross_val, args.tunning, args.gradient_tracking, not args.max_epoch, args.project)