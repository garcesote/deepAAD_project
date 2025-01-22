from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from utils.loss_functions import CustomLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

class CustomClassifier():

    def __init__(self, type, generation_model, criterion, batch_size = 128, shuffle = False, normalize = True):
        
        if type == 'LDA':
            self.classifier = LinearDiscriminantAnalysis()
        else:
            raise ValueError('Introduce a valid classifier')
        
        assert isinstance(generation_model, torch.nn.Module), 'The generation model must be a pytorch nn.Module model'
        self.model = generation_model

        assert isinstance(criterion, CustomLoss), 'The criterion must be a custom loss class'
        self.criterion = criterion

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.scaler = MinMaxScaler() if normalize else None

    # Compute the metrics from the train set to train the classifier
    def fit(self, train_set):

        metrics, labels = self._get_metrics(train_set)
        
        if self.normalize:
            metrics = self.scaler.fit_transform(metrics)

        # FIT THE CLASSIFIER
        self.classifier.fit(metrics, labels)

    def eval(self, test_set):

        metrics, labels = self._get_metrics(test_set)

        if self.normalize:
            metrics = self.scaler.fit_transform(metrics)

        loader_len = len(labels) // 2

        # EXTRACT THE SCORES / ACCURACIES
        att_scores = self.classifier.transform(metrics[:loader_len])
        unatt_scores = self.classifier.transform(metrics[loader_len:])
        att_correct = (att_scores > unatt_scores).sum()
        accuracy = (att_correct / loader_len) * 100
        print(f'Accuracy with {self.batch_size//64}s classifying is {accuracy:.4f}')
        
        accuracy_att = self.classifier.score(metrics[:loader_len], torch.ones((loader_len))) * 100
        accuracy_unatt = self.classifier.score(metrics[loader_len:], torch.zeros((loader_len))) * 100
        accuracies = [accuracy_att, accuracy_unatt, accuracy]

        return labels, metrics, accuracies

    def _get_metrics(self, dataset):

        loader = DataLoader(dataset, self.batch_size, shuffle = self.shuffle, pin_memory=True, drop_last=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # COLLECT THE PREDICTIONS OF THE MODEL
        corr_att_list = torch.zeros((len(loader)))
        corr_unatt_list = torch.zeros((len(loader)))
        ild_att_list = torch.zeros((len(loader)))
        ild_unatt_list = torch.zeros((len(loader)))

        train_loader_tqdm = tqdm(loader, desc='Computing model predictions to fit the classifier', leave=False, mininterval=0.5)
        with torch.no_grad():
            for batch, data in enumerate(train_loader_tqdm):
                
                eeg = data['eeg'].to(device, dtype=torch.float)
                stima = data['stima'].to(device, dtype=torch.float)
                stimb = data['stimb'].to(device, dtype=torch.float)
                
                y_hat = self.model(eeg)

                loss_list_att = self.criterion(preds=y_hat, targets = stima)
                corr_att_list[batch] = loss_list_att[1]
                ild_att_list[batch] = loss_list_att[2]

                loss_list_unatt = self.criterion(preds=y_hat, targets = stimb)
                corr_unatt_list[batch] = loss_list_unatt[1]
                ild_unatt_list[batch] = loss_list_unatt[2]

        # CONCATENATE AND NORMALIZE RESULTS
        corr_train = torch.cat((corr_att_list, corr_unatt_list))
        ild_train = torch.cat((ild_att_list, ild_unatt_list))
        metrics = torch.stack((corr_train, ild_train)).T
        labels = torch.cat((torch.ones(len(loader)), torch.zeros(len(loader))))

        return metrics, labels
        



