import torch
import torch.nn.functional as F
from utils.functional import get_mdl_name, multiple_loss_opt, set_seeds, get_data_path, get_channels, get_subjects, get_filename, get_loss
from utils.datasets import CustomDataset
from utils.loss_functions import CustomLoss
from utils.plot_results import plot_clsf_results
from torch.utils.data import DataLoader
from models.dnn import FCNN, CNN
from models.classifier import CustomClassifier
from models.vlaai import VLAAI
from models.eeg_conformer import Conformer, ConformerConfig
from statistics import mean
import numpy as np
import argparse
import yaml
import os
import json
import wandb
import sys

def main(config, wandb_upload, dataset, key, eval_population, finetuned, spatial_clsf, figures):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'spatial_audio'
    exp_name = config['exp_name']
    # window_list = [64, 128, 320, 640, 1280, 2560] # 1s, 2s, 5s, 10s, 20s, 40s
    window_list = [64, 128, 320, 640, 1600, 3200] # 1s, 2s, 5s, 10s, 25s, 50s
    # window_list = [3200]

    # spatial_clsf = True
    # figures = True
    # eval_population = True
    # finetuned = True
    
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

        if wandb_upload: wandb.init(project=project, name=exp_name, tags=[tag], config=run)

        window_accuracies = {win//64: None for win in window_list}

        # Config dataset
        ds_config = run['dataset_params']
        ds_config['leave_one_out'] = True if key == 'subj_independent' else False
        ds_config['hop'] = 1 if key == 'subj_independent' else False
        if ds_config['window_pred'] == False: ds_config['hop'] = 1 

        # Config training
        train_params = run['train_params']
        preproc_mode = train_params.get('preproc_mode')
        batch_size = train_params.get('batch_size')
        shuffle = train_params.get('shuffle')
        shuffle_test = shuffle if ds_config.get('window_pred') else False
        metrics_norm = train_params.get('metrics_norm', None)

        # Config loss
        loss_params = run['loss_params']
        loss_mode = loss_params.get('mode', 'mean')
        
        # If the loss mode only takes ILD into account and is finetunned, eval with both correlation and ils with alpha=0.1
        if loss_mode in ['ild_mae', 'ild_mse'] and finetuned:
            loss_params['mode'] = 'corr_' + loss_mode
            loss_params['alpha_end'] = 0.1

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # GET THE MODEL PATH
        mdl_load_path = os.path.join(global_path, 'results', project, key, mdl_load_folder)
        # If the finetunning loss is different from the original
        run['loss_params']['mode'] = run.get('finetune_loss', loss_mode)
        mdl_name = get_mdl_name(run)

        if finetuned:
            eval_path, dec_path = 'eval_finetuned_metrics', 'decode_finetuned_accuracy'  
        else:
            eval_path, dec_path = 'eval_metrics', 'decode_accuracy'

        # DEFINE THE SAVE PATH
        mdl_save_name = mdl_name + '_' + metrics_norm if metrics_norm is not None else mdl_name
        dst_save_path = os.path.join(global_path, 'results', project, key, eval_path, dataset+'_data', mdl_save_name)
        decAcc_save_path = os.path.join(global_path, 'results', project, key, dec_path, dataset+'_data', mdl_save_name)

        # Return to the original loss for evaluating
        run['loss_params']['mode'] = loss_mode

        eval_mean_results = []
        
        # Evaluate the models for each subject independently on the dataset
        if eval_population:
            selected_subjects = [get_subjects(dataset)]
        else:
            selected_subjects = get_subjects(dataset)

        # Define the time-shift for computing the null-distribution
        time_shift = 100

        for eval_window in window_list:

            eval_results = {}
            nd_results = {} # construct a null distribution when evaluating
            dec_results = []

            for subj in selected_subjects:
                
                subj_name ='population' if eval_population else subj
                str_win = str(eval_window//64)+'s' if 'VLAAI' in model else str(batch_size//64)+'s'

                print(f'Evaluating {model} on window {eval_window//64}s with {dataset} dataset for subj {subj_name}')

                if loss_mode is not None: print(f'Evaluating the network based on {loss_mode} criterion')

                mdl_folder = os.path.join(mdl_load_path, dataset+'_data', mdl_name)
                if key == 'population' and not finetuned:
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

                # LOAD THE DATA
                data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
                test_set = CustomDataset(
                    dataset=dataset,
                    data_path=data_path,
                    split='test',
                    subjects=subj,
                    **ds_config
                )

                # LOSS FUNCTION
                criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

                # WHEN MULTIPLE METRICS OPTIMIZED USE A LINEAR CLASSIFIER
                if spatial_clsf and multiple_loss_opt(loss_mode):
                    
                    classifier = CustomClassifier('LDA', mdl, criterion, batch_size=eval_window, shuffle=shuffle_test, normalize=metrics_norm)

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
                    labels, metrics, accuracies = classifier.eval(test_set= test_set)

                    # SAVE RESULTS FOT MESD
                    dec_results.append(accuracies[2])

                    # SAVE FIGURES
                    if figures:
                        clsf_name = 'LDA_trial' if not finetuned else 'LDA_finetuned'
                        save_figures_path = os.path.join(global_path, 'figures', project, clsf_name, mdl_save_name) 
                        save_figures_path =  os.path.join(save_figures_path, 'population') if isinstance(subj, list) else os.path.join(save_figures_path, subj) 
                        plot_clsf_results(classifier.classifier, metrics, labels, eval_window, accuracies, save_figures_path)

                else:

                    test_loader = DataLoader(test_set, batch_size=eval_window, shuffle= shuffle_test, pin_memory=True)
                    
                    # EVALUATE THE MODEL
                    eval_loss = []
                    eval_nd_loss = []
                    eval_loss_list = []
                    
                    att_corr = 0
                    with torch.no_grad():
                        for batch, data in enumerate(test_loader):
                            
                            eeg = data['eeg'].to(device, dtype=torch.float)
                            stima = data['stima'].to(device, dtype=torch.float)
                            
                            preds = mdl(eeg)

                            loss_list = criterion(preds=preds, targets = stima)
                            loss = loss_list[0]

                            # Calculates Pearson's coef. for the matching distribution and for the null one
                            nd_loss = criterion(preds=preds, targets = torch.roll(stima, time_shift))[0]

                            if data['stimb'] is not None:
                                if loss_mode != 'spatial_locus':
                                    stimb = data['stimb'].to(device, dtype=torch.float)
                                    # Decoding accuracy of the model
                                    unat_loss_list = criterion(preds=preds, targets = stimb)
                                    unat_loss = unat_loss_list[0]
                                    if loss.item() < unat_loss.item():
                                        att_corr += 1                                    
                                else:
                                    probs = F.sigmoid(preds)
                                    pred_labels = (probs >= 0.5).float()
                                    n_correct = torch.sum(pred_labels == stima)
                                    if n_correct > batch_size // 2:
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
    parser.add_argument("--config", type=str, default='configs/finetunning/ild_penalty.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--finetuned", action='store_true', help="When included search for the model on the finetune folder")
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

    main(config, wandb_upload, args.dataset, args.key, args.eval_population, args.finetuned, args.spatial_clsf, args.figures)