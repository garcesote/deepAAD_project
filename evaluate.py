import torch
import torch.nn.functional as F

from utils.functional import verbose, load_model, get_mdl_name, multiple_loss_opt, set_seeds, get_data_path, get_subjects, get_filename
from utils.datasets import CustomDataset
from utils.loss_functions import CustomLoss
from utils.plot_results import plot_clsf_results
from torch.utils.data import DataLoader
from models.classifier import CustomClassifier
from statistics import mean

import numpy as np
import argparse
import yaml
import os
import json
import wandb
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(config, wandb_upload, dataset, key, cross_val, eval_population, finetuned, spatial_clsf, figures):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'euroacustics'
    exp_name = config['exp_name']
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
        mdl_load_folder = 'finetune_models' if finetuned else 'models'

        tag = 'evaluation_finetune' if finetuned else 'evaluation'
        tag = tag + '_spatial_clsf' if spatial_clsf else tag

        # Config dataset
        ds_config = run['dataset_params']
        ds_config['leave_one_out'] = True if key == 'subj_independent' else False
        # ds_config['hop'] = 1 if key == 'subj_independent' else False
        if ds_config['window_pred'] == False: ds_config['hop'] = 1
        stim_input = ds_config.get('stim_input', False)  

        # Config training
        train_params = run['train_params']
        preproc_mode = train_params.get('preproc_mode')
        shuffle = train_params.get('shuffle')
        shuffle_test = shuffle if ds_config.get('window_pred') else False
        metrics_norm = train_params.get('metrics_norm', None)

        # Config loss
        loss_params = run['loss_params']
        loss_mode = loss_params.get('mode', 'mean')

        # If the finetunning loss is different from the original (used for saving)
        run['loss_params']['mode'] = run.get('finetune_loss', loss_mode)
        
        # Define the model_name
        mdl_name = get_mdl_name(run)
        
        # If the loss mode only takes ILD into account and is finetunned, eval with both correlation and ils with alpha=0.1
        if loss_mode in ['ild_mae', 'ild_mse'] and finetuned:
            loss_params['mode'] = 'corr_' + loss_mode
            loss_params['alpha_end'] = 0.1

        eval_mean_results = []
        
        # Evaluate the models for each subject independently on the dataset
        if eval_population:
            selected_subjects = [get_subjects(dataset)]
        else:
            selected_subjects = get_subjects(dataset)

        # Define the time-shift for computing the null-distribution
        time_shift = 100

        # Cross validation 
        if cross_val:
            n_folds = 5
        else:
            n_folds = 1

        for cv_fold in range(n_folds):
            
            if cross_val: run['cv_fold'] = cv_fold
            run['key'] = key
            run['dataset'] = dataset
            if wandb_upload: wandb.init(project=project, name=exp_name, tags=[tag], config=run)
            if not cross_val: cv_fold = None

            window_accuracies = {win//64: None for win in window_list}

            for eval_window in window_list:

                eval_results = {}
                nd_results = {} # construct a null distribution when evaluating
                dec_results = []

                # For window pred (1, eval_window, T) and for sample pred (eval_window, window, T)
                batch_size = 1 if ds_config.get('window_pred') else eval_window
                if ds_config.get('window_pred'): ds_config['window'] = eval_window

                for subj in selected_subjects:
                    
                    subj_name ='population' if eval_population else subj
                    str_win = str(eval_window//64)+'s'

                    verbose('evaluate', key, subj, dataset, model, eval_window//64, loss_mode, cv_fold=cv_fold)

                    if finetuned:
                        eval_path, dec_path = 'eval_finetuned_metrics', 'decode_finetuned_accuracy'  
                    else:
                        eval_path, dec_path = 'eval_metrics', 'decode_accuracy'

                    # DEFINE THE SAVE PATH
                    mdl_save_name = mdl_name + '_' + metrics_norm if metrics_norm is not None else mdl_name
                    cv_folder = f'cvFold={cv_fold}' if cross_val else 'global_validation'
                    dst_save_path = os.path.join(global_path, 'results', project, key, eval_path, dataset+'_data', mdl_save_name, cv_folder)
                    decAcc_save_path = os.path.join(global_path, 'results', project, key, dec_path, dataset+'_data', mdl_save_name, cv_folder)

                    # GET THE MODEL LOAD PATH
                    mdl_load_path = os.path.join(global_path, 'results', project, key, mdl_load_folder, dataset+'_data', mdl_name)
                    if key != 'population' or finetuned:
                        mdl_load_path = os.path.join(mdl_load_path, subj)

                    mdl_filename = get_filename(mdl_load_path, cv_fold)
                    mdl_load_path = os.path.join(mdl_load_path, mdl_filename)

                    # Return to the original loss for evaluating
                    run['loss_params']['mode'] = loss_mode
                    
                    # LOAD THE MODEL
                    mdl = load_model(run, dataset, wandb_upload)
                    mdl.load_state_dict(torch.load(mdl_load_path, map_location=torch.device(device), weights_only=True))
                    mdl.to(device)

                    # LOAD THE DATA
                    data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
                    test_set = CustomDataset(
                        dataset=dataset,
                        data_path=data_path,
                        split='test',
                        subjects=subj,
                        cv_fold=cv_fold,
                        **ds_config
                    )

                    # LOSS FUNCTION
                    criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

                    # WHEN MULTIPLE METRICS OPTIMIZED USE A LINEAR CLASSIFIER
                    if spatial_clsf and multiple_loss_opt(loss_mode):
                        
                        classifier = CustomClassifier('LDA', mdl, criterion, batch_size=batch_size, shuffle=shuffle_test, normalize=metrics_norm)

                        # LOAD TRAIN DATA AND FIT CLASSIFIER
                        train_set = CustomDataset(
                            dataset=dataset,
                            data_path=data_path,
                            split='train',
                            subjects=subj,
                            **ds_config
                        )
                        classifier.fit(train_set=train_set)

                        # EVALUATE THE CLASSIFIER WITH TEST SET
                        labels, metrics, accuracies = classifier.eval(test_set=test_set)

                        # SAVE RESULTS FOT MESD
                        dec_results.append(accuracies[2])

                        # SAVE FIGURES
                        if figures:
                            clsf_name = 'LDA_trial' if not finetuned else 'LDA_finetuned'
                            save_figures_path = os.path.join(global_path, 'figures', project, clsf_name, mdl_save_name) 
                            save_figures_path =  os.path.join(save_figures_path, 'population') if isinstance(subj, list) else os.path.join(save_figures_path, subj) 
                            plot_clsf_results(classifier.classifier, metrics, labels, eval_window, accuracies, save_figures_path)

                    # NO NEED TO CLASSIFY WHEN ONE METRIC IT'S EMPLOYED
                    else:

                        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle= shuffle_test, pin_memory=True)
                        
                        # EVALUATE THE MODEL
                        eval_loss = []
                        eval_nd_loss = []
                        eval_loss_list = []
                        
                        att_corr = 0
                        with torch.no_grad():
                            for batch, data in enumerate(test_loader):
                                
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

                                # Calculates Pearson's coef. for the matching distribution and for the null one
                                nd_loss = criterion(preds=preds, targets=torch.roll(targets, time_shift))[0]

                                # Decode accuracy
                                if dataset != 'skl':
                                    # Direct classification based on the predictions
                                    if loss_mode == 'spatial_locus':
                                        probs = F.sigmoid(preds)
                                        pred_labels = (probs >= 0.5).float()
                                        n_correct = torch.sum(pred_labels == stima).float()
                                        if n_correct > batch_size / 2:
                                            att_corr += 1
                                    elif stim_input:
                                        pred_labels = (preds >= 0.5).float()
                                        n_correct = torch.sum(pred_labels == targets).float()
                                        if n_correct > batch_size / 2:
                                            att_corr += 1

                                    # Classification by comparing loss
                                    else:
                                        unat_loss_list = criterion(preds=preds, targets = stimb)
                                        unat_loss = unat_loss_list[0]
                                        if loss.item() < unat_loss.item():
                                            att_corr += 1

                                # Append all losses for eval results
                                eval_loss.append(loss.item())
                                eval_loss_list.append(loss_list)
                                eval_nd_loss.append(nd_loss.item())

                        eval_results[subj_name] = eval_loss
                        nd_results[subj_name] = eval_nd_loss
                        dec_accuracy = (att_corr / len(test_loader)) * 100
                        dec_results.append(dec_accuracy)
                        
                        if multiple_loss_opt(loss_mode):
                            eval_mean_results.append([mean(eval_loss), 
                                torch.mean(torch.hstack([loss_list[1] for loss_list in eval_loss_list])).item(), 
                                torch.mean(torch.hstack([loss_list[2] for loss_list in eval_loss_list])).item()])
                        else:
                            eval_mean_results.append([mean(eval_loss)])

                        print(f'Subject {subj_name} | corr_mean {mean(eval_loss)} | decode_accuracy {dec_accuracy}')

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

                # UPLOAD RESULTS TO WANDB
                if wandb_upload:
                    wandb_log = {'window': eval_window, 
                        'decAcc_subj_mean': np.mean(dec_results), 
                        'decAcc_subj_std': np.std(dec_results)
                    }
                    # Upload the loss when not using the LDA
                    if not spatial_clsf:
                        loss_mean_results = [results[0] for results in eval_mean_results]
                        wandb_log = {'window': eval_window, 
                            'loss_subj_mean': np.mean(loss_mean_results), 
                            'loss_subj_std': np.std(loss_mean_results), 
                            'decAcc_subj_mean': np.mean(dec_results), 
                            'decAcc_subj_std': np.std(dec_results)
                        }
                        # Add isolated metrics for the log when the loss is computed by multiple criterion (correlation + ild)
                        if multiple_loss_opt(loss_mode):
                            wandb_log['corr_subj_mean'] = np.mean([results[1] for results in eval_mean_results])
                            wandb_log['ild_subj_mean'] = np.mean([results[2] for results in eval_mean_results])
                            wandb_log['corr_subj_std'] = np.std([results[1] for results in eval_mean_results])
                            wandb_log['ild_subj_std'] = np.std([results[2] for results in eval_mean_results])
                    
                    wandb.log(wandb_log)

                # Save the window results to compute mesd
                window_accuracies[eval_window//64] = dec_results

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
    parser.add_argument("--config", type=str, default='configs/stim_input/fast_aad_net.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--finetuned", action='store_true', help="When included search for the model on the finetune folder")
    parser.add_argument("--cross_val", action='store_true', help="When included select the cross validation models")    
    parser.add_argument("--eval_population", action='store_true', help="When included evaluate by fitting a classifier with both")    
    parser.add_argument("--spatial_clsf", action='store_true', help="When included evaluate by fitting a classifier with both")
    parser.add_argument("--figures", action='store_true', help="When included evaluate by fitting a classifier with both")

    args = parser.parse_args()

    wandb_upload = args.wandb

    # Introduce path to the mesd-toolbox
    sys.path.append("C:/Users/jaulab/Desktop/AAD/mesd-toolbox/mesd-toolbox-python")
    # sys.path.append("C:/Users/garce/Desktop/proyecto_2024/mesd-toolbox/mesd-toolbox-python")
    from mesd_toolbox import compute_MESD
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload, args.dataset, args.key, args.cross_val, args.eval_population, args.finetuned, args.spatial_clsf, args.figures)