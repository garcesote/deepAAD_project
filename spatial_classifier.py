import torch
from torch.utils.data import DataLoader
import os
import sys
import yaml
import wandb
import json
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

from models.dnn import FCNN, CNN
from utils.datasets import CustomDataset
from utils.loss_functions import CustomLoss
from utils.functional import get_mdl_name, set_seeds, get_subjects, get_channels, get_data_path, get_filename
from utils.plot_results import plot_clsf_results

def main(config, wandb_upload, dataset, key, finetuned, save_figures):

    global_path = config['global_path']
    global_data_path = config['global_data_path']
    project = 'spatial_audio'
    config['dataset'] = dataset
    config['key'] = key

    window_list = [64, 128, 320, 640, 1600, 3200] # 1s, 2s, 5s, 10s, 25s, 50s

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

        mdl_load_folder = 'finetune_models' if finetuned else 'models'
        mdl_load_path = os.path.join(global_path, 'results', project, key, mdl_load_folder)

        if wandb_upload: wandb.init(project=project, name=exp_name, tags=['spatial_classifier'], config=run)

        window_accuracies = {win//64: None for win in window_list}

        if key == 'population':
            selected_subj = [get_subjects(dataset)]
        else:
            selected_subj = get_subjects(dataset)

        # Config dataset
        ds_config = run['dataset_params']
        ds_config['leave_one_out'] = True if key == 'subj_independent' else False

        train_params = run['train_params']
        preproc_mode = ds_config['preproc_mode'] if 'preproc_mode' in ds_config.keys() else None
        window_pred = ds_config.get('window_pred')
        ds_config['hop'] = ds_config['hop'] if window_pred else 1
        data_path = get_data_path(global_data_path, dataset, preproc_mode = preproc_mode)

        for eval_window in window_list:
            
            fixed = ds_config['fixed']
            batch_size = eval_window
            shuffle = ds_config['shuffle'] if window_pred else False

            # GET THE MODEL PATH
            mdl_name = get_mdl_name(run)

            dataset_name = dataset+'_fixed' if fixed else dataset

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            for subj in selected_subj:

                # Load the sets (only for population evaluation)
                data_path = get_data_path(global_data_path, dataset, preproc_mode=preproc_mode)
                train_set = CustomDataset(
                    dataset=dataset,
                    data_path=data_path,
                    split='train',
                    subjects=subj,
                    **ds_config
                )
                test_set = CustomDataset(
                    dataset=dataset,
                    data_path=data_path,
                    split='test',
                    subjects=subj,
                    **ds_config
                )

                if key == 'population':
                    print(f'Fitting classifier for {model} on all subjects with {dataset} data...')
                else:
                    run['subj'] = subj
                    if key == 'subj_independent':
                        print(f'Fitting classifier for  {model} leaving out {subj} with {dataset} data...')
                    else:
                        print(f'Fitting classifier for  {model} on {subj} with {dataset} data...')
                

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
                else:
                    raise ValueError('Introduce a valid model')
                mdl.load_state_dict(torch.load(os.path.join(mdl_folder, mdl_filename), map_location=torch.device(device)))
                mdl.to(device)
                mdl.eval()
                
                # LOSS FUNCTION
                criterion = CustomLoss(window_pred=ds_config.get('window_pred', True), **run['loss_params'])

                # LOAD DATA
                train_loader = DataLoader(train_set, batch_size, shuffle= shuffle, pin_memory=True)
                
                # COLLECT THE PREDICTIONS OF THE MODEL
                corr_att_list = torch.zeros((len(train_loader)))
                corr_unatt_list = torch.zeros((len(train_loader)))
                ild_att_list = torch.zeros((len(train_loader)))
                ild_unatt_list = torch.zeros((len(train_loader)))

                train_loader_tqdm = tqdm(train_loader, desc='Computing model predictions to fit the classifier', leave=False, mininterval=0.5)
                with torch.no_grad():
                    for batch, data in enumerate(train_loader_tqdm):
                        
                        eeg = data['eeg'].to(device, dtype=torch.float)
                        stima = data['stima'].to(device, dtype=torch.float)
                        stimb = data['stimb'].to(device, dtype=torch.float)
                        
                        y_hat = mdl(eeg)

                        loss_list_att = criterion(preds=y_hat, targets = stima)
                        corr_att_list[batch] = loss_list_att[1]
                        ild_att_list[batch] = loss_list_att[2]

                        loss_list_unatt = criterion(preds=y_hat, targets = stimb)
                        corr_unatt_list[batch] = loss_list_unatt[1]
                        ild_unatt_list[batch] = loss_list_unatt[2]

                # CONCATENATE AND NORMALIZE RESULTS
                corr_train = torch.cat((corr_att_list, corr_unatt_list))
                ild_train = torch.cat((ild_att_list, ild_unatt_list))
                metrics = torch.stack((corr_train, ild_train)).T
                labels = torch.cat((torch.ones(len(train_loader)), torch.zeros(len(train_loader))))

                # NORMALIZATION
                scaler = MinMaxScaler()
                norm_metrics = scaler.fit_transform(metrics)

                # FIT THE CLASSIFIER
                spatial_clsf = LinearDiscriminantAnalysis()
                spatial_clsf.fit(norm_metrics, labels)

                # EVAL RESULTS
                test_loader = DataLoader(test_set, batch_size = batch_size, shuffle= False, pin_memory=True)

                # COLLECT THE VALIDATION PREDICTIONS OF THE MODEL
                corr_att_list = torch.zeros((len(test_loader)))
                corr_unatt_list = torch.zeros((len(test_loader)))
                ild_att_list = torch.zeros((len(test_loader)))
                ild_unatt_list = torch.zeros((len(test_loader)))

                att_correct = 0
                test_loader_tqdm = tqdm(test_loader, desc='Computing model predictions to evaluate the classifier', leave=False, mininterval=0.5)
                with torch.no_grad():
                    for batch, data in enumerate(test_loader_tqdm):
                        
                        eeg = data['eeg'].to(device, dtype=torch.float)
                        stima = data['stima'].to(device, dtype=torch.float)
                        stimb = data['stimb'].to(device, dtype=torch.float)
                        
                        y_hat = mdl(eeg)

                        [loss_att, corr_att, ild_att] = criterion(preds=y_hat, targets = stima)
                        corr_att_list[batch] = corr_att
                        ild_att_list[batch] = ild_att

                        [loss_unatt, corr_unatt, ild_unatt] = criterion(preds=y_hat, targets = stimb)
                        corr_unatt_list[batch] = corr_unatt
                        ild_unatt_list[batch] = ild_unatt
                        
                        corr_att, corr_unatt, ild_att, ild_unatt = corr_att.cpu(), corr_unatt.cpu(), ild_att.cpu(), ild_unatt.cpu()
                        score_a = spatial_clsf.transform(torch.stack((corr_att, ild_att)).unsqueeze(0))
                        score_b = spatial_clsf.transform(torch.stack((corr_unatt, ild_att)).unsqueeze(0))

                        if score_a >= score_b:
                            att_correct += 1

                accuracy = (att_correct / len(test_loader)) * 100
                print(f'Accuracy with {batch_size//64}s classifying is {accuracy:.4f}')

                # CONCATENATE AND NORM TO FEED THE CLASSIFIER
                corr_val = torch.cat((corr_att_list, corr_unatt_list))
                ild_val = torch.cat((ild_att_list, ild_unatt_list))
                metrics = torch.stack((corr_val, ild_val)).T
                labels = torch.cat((torch.ones(len(test_loader)), torch.zeros(len(test_loader))))

                scaler = MinMaxScaler()
                norm_metrics = scaler.fit_transform(metrics)

                # EXTRACT THE ATTENDED AND UNATTENDED ACCURACIES
                accuracy_att = spatial_clsf.score(norm_metrics[:len(test_loader)], torch.ones((len(test_loader))))
                accuracy_unatt = spatial_clsf.score(norm_metrics[len(test_loader):], torch.zeros((len(test_loader))))
                accuracies = [accuracy_att, accuracy_unatt, accuracy]
                
                # Save the window results to compute mesd
                window_accuracies[eval_window//64] = accuracy                

                # LOG RESULTS
                wandb_log = {'window': eval_window, 'accuracy': accuracy, 'accuracy_att': accuracy_att, 'accuracy_unattended': accuracy_unatt}
                if wandb_upload: wandb.log(wandb_log)

                # SAVE FIGURES
                if save_figures:
                    save_figures_path = os.path.join(global_path, 'figures', project, 'LDA_Classifiers', mdl_name)
                    plot_clsf_results(spatial_clsf, norm_metrics, labels, eval_window, accuracies, save_figures_path)

        # COMPUTE MESD (default values)
        print('Computing MESD')
        mesd_dict = {'mesd': None, 'N_mesd': None, 'tau_mesd': None, 'p_mesd': None}
        tau = np.array(list(window_accuracies.keys()))
        p = np.array([results / 100 for results in window_accuracies.values()])
        if np.any(p > 0.5):
            mesd, N_mesd, tau_mesd, p_mesd = compute_MESD(tau,p)
            print(f"The minimal expected switch duration is MESD = {mesd} \nat an otpimal working point of (tau, p) = ({tau_mesd}, {p_mesd}) \nwith N = {N_mesd} states in the Markov chain.")
        else:
            mesd, N_mesd, tau_mesd, p_mesd = 0, 0, 0, 0
            print(f"The MESD could not be computed")
        mesd_results = [mesd, N_mesd, tau_mesd, p_mesd]
        for idx, value in zip(list(mesd_dict.keys()), mesd_results):
            mesd_dict[idx] = value
        if wandb_upload: wandb.log(mesd_dict)

        if wandb_upload: wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Add config argument
    parser.add_argument("--config", type=str, default='configs/spatial_audio/ild_best_models.yaml', help="Ruta al archivo config")
    parser.add_argument("--wandb", action='store_true', help="When included actualize wandb cloud")
    parser.add_argument("--figure", action='store_true', help="When included generate and save the figures of the classifier")    
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--key", type=str, default='population', help="Key from subj_specific, subj_independent and population")
    parser.add_argument("--finetuned", action='store_true', help="When included search for the model on the finetune folder")

    args = parser.parse_args()

    wandb_upload = args.wandb

    # Introduce path to the mesd-toolbox
    sys.path.append(r"C:\Users\jaulab\Desktop\AAD\mesd-toolbox\mesd-toolbox-python")
    from mesd_toolbox import compute_MESD
    
    # Upload results to wandb
    if wandb_upload:
        wandb.login()

    # Load corresponding config
    with open(args.config, 'r') as archivo:
        # Llamar a la función de entrenamiento con los argumentos
        config = yaml.safe_load(archivo)

    main(config, wandb_upload, args.dataset, args.key, args.finetuned, args.figure)
            