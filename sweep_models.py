from tqdm import tqdm
from utils.loss_functions import CustomLoss
import yaml
import wandb

from utils.functional import verbose, load_model, multiple_loss_opt, get_data_path, get_subjects, set_seeds
from utils.datasets import CustomDataset
from utils.sampler import BatchRandomSampler

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_training_run(run, config, dataset, global_data_path, project, key, cv_fold, early_stop):

    # Global params
    model = run['model']

    # Get configs
    train_config = run['train_params']
    ds_config = run['dataset_params']
    loss_config = run['loss_params']

    # Config training
    lr = float(train_config['lr'])
    max_epoch = train_config.get('max_epoch', 200)
    scheduler_patience = train_config.get('scheduler_patience', 10)
    early_stopping_patience = train_config['early_stopping_patience'] if early_stop else max_epoch
    batch_rnd_sampler = train_config.get('batch_rnd_sampler', False)
    preproc_mode = train_config.get('preproc_mode')
    shuffle = train_config.get('shuffle', False)
    val_shuffle = shuffle if ds_config.get('window_pred') else False

    # Config dataset
    ds_config['leave_one_out'] = True if key == 'subj_independent' else False
    ds_val_config = ds_config.copy()
    if ds_config['window_pred'] == False: ds_val_config['hop'] = 1
    stim_input = ds_config.get('stim_input', False) 

    # Config loss
    loss_mode = loss_config.get('mode', 'mean')
    
    # Population mode that generates a model for all samples
    if key == 'population':
        selected_subj = [get_subjects(dataset)]
    else:
        selected_subj = get_subjects(dataset)

    for subj in selected_subj:
        
        # VERBOSE
        verbose('train', key, subj, dataset, model, loss_mode=loss_mode, cv_fold=cv_fold)
        
        # WANDB INIT
        run['subject'] = subj
        run['cv_fold'] = cv_fold
        run['key'] = key
        run['dataset'] = dataset
        wandb.init(project=project, tags=['sweep'], config=run)

        # LOAD THE MODEL
        mdl = load_model(run, dataset, wandb_upload=True, sweep=True)
        mdl.to(device)
        mdl_size = sum(p.numel() for p in mdl.parameters())

        # WEIGHT INITILAIZATION
        init_weights = train_config.get('init_weights', False)
        if init_weights: mdl.init_weights()
        
        # Sweep training params
        batch_size = getattr(wandb.config, 'batch_size', train_config['batch_size'])
        weight_decay = float(getattr(wandb.config, 'weight_decay', train_config.get('weight_decay', 1e-8)))
        
        # Val batch size
        val_batch_size = 1 if ds_config.get('window_pred') else batch_size
        # Exception on val_batch_size on window pred when using triplet loss
        if loss_mode == 'triplet_loss': val_batch_size = batch_size

        # LOAD THE DATA
        data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
        train_set = CustomDataset(
            dataset=dataset,
            data_path=data_path,
            split='train',
            subjects=subj,
            cv_fold = None,
            **ds_config
        )
        val_set = CustomDataset(
            dataset=dataset,
            data_path=data_path,
            split='val',
            subjects=subj,
            cv_fold = None,
            **ds_val_config
        )

        if batch_rnd_sampler:
            batch_sampler = BatchRandomSampler(train_set, batch_size)
            train_loader = DataLoader(train_set, batch_sampler=batch_sampler, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size, shuffle = shuffle, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, val_batch_size, shuffle= val_shuffle, pin_memory=True)

        # OPTIMIZER PARAMS
        optimizer = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

        # LOSS FUNCTION
        criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

        # Early stopping parameters
        best_epoch = 0
        best_loss = 0
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

            mdl.eval()
            val_loss = []
            val_att_corr = 0

            # Validation
            with torch.no_grad():

                for batch, data in enumerate(val_loader):

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
                                val_att_corr += 1
                        elif stim_input:
                            pred_labels = (preds >= 0.5).float()
                            n_correct = torch.sum(pred_labels == targets).float()
                            if n_correct > val_batch_size / 2:
                                val_att_corr += 1

                        # Classification by comparing loss
                        else:
                            unat_loss_list = criterion(preds=preds, targets = stimb)
                            unat_loss = unat_loss_list[0]
                            if loss.item() < unat_loss.item():
                                val_att_corr += 1
                            # When multiple loss compute the accuracy for each metric
                            if len(loss_list) > 1:
                                if loss_list[1].item() < unat_loss_list[1].item():
                                    val_att_corr += 1
                                if loss_list[2].item() < unat_loss_list[2].item():
                                    val_att_corr += 1

                    val_loss.append(loss_list)

            mean_val_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in val_loss])).item()
            mean_train_loss = torch.mean(torch.hstack([loss_list[0] for loss_list in train_loss])).item()
            val_decAccuracy = val_att_corr / len(val_loader) * 100
            val_corr_decAccuracy = val_att_corr / len(val_loader) * 100
            val_ild_decAccuracy = val_att_corr / len(val_loader) * 100

            scheduler.step(-mean_val_loss)

            # Logging metrics
            if multiple_loss_opt(loss_mode):
                print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f} | Val ild_acc: {val_ild_decAccuracy} | Val corr_acc: {val_corr_decAccuracy}')
            else:
                print(f'Epoch: {epoch} | Train loss: {mean_train_loss:.4f} | Val loss/acc: {mean_val_loss:.4f}/{val_decAccuracy:.4f}')
            
            wandb_log = {'mdl_size': mdl_size, 'train_loss': mean_train_loss, 'val_loss': mean_val_loss, 'val_acc': val_decAccuracy}
            # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
            if multiple_loss_opt(loss_mode):
                wandb_log['val_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in val_loss])).item()
                wandb_log['train_corr'] = torch.mean(torch.hstack([loss_list[1] for loss_list in train_loss])).item()
                wandb_log['val_acc_corr'] = val_corr_decAccuracy
                wandb_log['val_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in val_loss])).item()
                wandb_log['train_ild'] = torch.mean(torch.hstack([loss_list[2] for loss_list in train_loss])).item()
                wandb_log['val_acc_ild'] = val_ild_decAccuracy
            wandb.log(wandb_log)

            # Save best results
            if mean_val_loss < best_loss or epoch == 0:
                best_loss = mean_val_loss
                best_accuracy = val_decAccuracy
                best_epoch = epoch

        wandb.log({'best_epoch': best_epoch, 'best_loss': best_loss, 'best_acc':best_accuracy})
        wandb.finish()

def main(config, dataset, key):

    global_data_path = config['global_data_path']
    project = 'euroacustics'
    config['dataset'] = dataset
    config['key'] = key
    cross_val = None

    # REPRODUCIBILITY
    if 'seed' in config.keys(): 
        set_seeds(config['seed'])
        exp_name =  exp_name + '_' + config['seed']
    else: 
        set_seeds() # default seed = 42

    for run in config['runs']:
        process_training_run(run, config, dataset, global_data_path, project, key, cross_val, early_stop=True)

if __name__ == "__main__":

    config_path = 'configs/euroacustics/conformer.yaml'
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
