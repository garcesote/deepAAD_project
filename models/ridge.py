from scipy.linalg import toeplitz
from scipy.stats import pearsonr
from collections.abc import Iterable
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge as Ridge_sklean

class Ridge:
    def __init__(self, start_lag=0, end_lag=50, alpha=1, trial_len=3200, verbose=True, original =False, per_trial:bool = False):
        '''
        num_lags: how many latencies to consider for the system response
        offset: when does the system response begin wrt an impulse timestamp? (may be negative)
        alpha: the regularisation parameter(s).
        trial_len: samples for each trial of the ds
        '''
        self.start_lag=start_lag
        self.end_lag = end_lag
        self.num_lags = self.end_lag-self.start_lag
        self.original = original
        self.trial_len = trial_len
        if self.end_lag>0 and self.start_lag<0:
            self.num_lags+=1
        self.per_trial = per_trial
        self.best_alpha_idx=False
        
        if isinstance(alpha, Iterable):
            self.alphas = alpha
        else:
            self.alphas = np.array([alpha])
            self.best_alpha_idx=0

        self.verbose = verbose

        
    def fit(self, X, y):
        '''
        inputs:
        - X, ndarray of shape (n_times, n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        '''
        
        # 1. Check that data shapes make sense
        if self.verbose:
            print("Checking inputs...")
        
        n_times, self.n_input_features = X.shape
        n_output_times, self.n_output_features = y.shape
        assert(n_times==n_output_times)
        
        # 2. Form the circulant data matrix
        lagged_matrix = np.empty((self.num_lags*self.n_input_features, n_times))
        if self.per_trial:
            n_trials = n_times // self.trial_len
            for n in tqdm(range(n_trials), desc="Computing lagged matrix"):
                start = n * self.trial_len
                end = start + self.trial_len
                lagged_matrix[:, start:end] = self._get_lagged_matrix(X[start:end, :].T)
        else:
            lagged_matrix = self._get_lagged_matrix(X.T)

        lagged_matrix = lagged_matrix.T

        if self.verbose:
            print('Computing autocorr_matrix...')
        XtX = np.dot(lagged_matrix.T, lagged_matrix)
        
        # 3. Perform Ridge
        S, V = np.linalg.eigh(XtX)

        # Sort the eigenvalues
        s_ind = np.argsort(S)[::-1]
        S = S[s_ind]
        V = V[:, s_ind] 

        # optional pcr stage

        # # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
        # # and compute the average
        # tol = np.finfo(float).eps
        # r = sum(S > tol)
        # #S = S[0:r]
        # #V = V[:, 0:r]
        # nl = np.mean(S)
        
        # 4. Apply ridge regression
        if self.verbose:
            print("Calculating coefficients...")

        self.coef_ = np.empty((self.alphas.size, self.n_output_features, self.n_input_features, self.num_lags))
        for i, alpha in tqdm(enumerate(self.alphas), desc='Itenrating through alphas'):
            for j in range(self.n_output_features):
                XtY = np.dot(lagged_matrix.T, y[:, j])
                if not self.original:
                    z = np.dot(V.T, XtY)
                    tmp_coefs = V @ np.diag(1/(S+alpha)) @ z[:, np.newaxis]
                else:
                    z = np.mean(np.diag(XtX))
                    Q = np.eye(XtX.shape[0], XtX.shape[1])
                    rdg = alpha * z * Q
                    tmp_coefs =  np.linalg.inv(XtX + rdg) @ XtY[:, np.newaxis]
                self.coef_[i, j, :, :] = np.reshape(tmp_coefs[:, 0], (self.num_lags, self.n_input_features)).T


    def predict(self, X, best_alpha=True):

        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - best_alpha: whether to make predictions for all regularisation parameters, or just the best one
        returns:
        - preditions, ndarray of shape (n_alphas, n_output_features, n_times)
        '''
        
        n_times, n_input_features = X.shape
        assert n_input_features == self.n_input_features, f'Input must have {self.n_input_features} channels'

        # 1. Compute the lagged matrix
        lagged_matrix = self._get_lagged_matrix(X.T)
        
        lagged_matrix = lagged_matrix.T # (L*C, T) => (T, L*C)
        
        # 2. Create predictions for every alpha and every output feature
        if best_alpha == False:
            
            predictions = np.empty((self.alphas.size, self.n_output_features, n_times))
            
            for i, alpha in enumerate(self.alphas):
                for j in range(self.n_output_features):
                    preds = lagged_matrix @ self.coef_[i, j].T.reshape(self.n_input_features*self.num_lags, 1)
                    predictions[i,j] = preds.flatten()
                    
        else:
            
            predictions = np.empty((self.n_output_features, n_times))
            for j in range(self.n_output_features):
                preds = lagged_matrix @ self.coef_[self.best_alpha_idx, j].T.reshape(self.num_lags*self.n_input_features, 1)
                predictions[j] = preds.flatten()

        return predictions
    
    
    def score(self, X, y, best_alpha=True, pad=False):

        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        - best_alpha: whether to score for best reg or all regularisation parameters
        - pad: whether to make predictions of the same size as y
        returns:
        - scores, ndarray of shape (n_alphas, n_output_features)
        '''
        
        predictions = self.predict(X, best_alpha=best_alpha)
        
        if best_alpha==False:
        
            scores = np.empty((self.alphas.size, self.n_output_features))
            for i, alpha in enumerate(self.alphas):
                for j in range(self.n_output_features):
                    scores[i, j] = pearsonr(predictions[i,j], y[:, j])[0]
        
        else:
            scores = np.empty((self.n_output_features))
            for j in range(self.n_output_features):
                scores[j] = pearsonr(predictions[j], y[:, j])[0]
            
        return scores

    
    def score_in_batches(self, X, y, batch_size=125):

        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        returns:
        - scores, ndarray of shape (n_alphas, n_output_features)
        '''
        
        predictions = self.predict(X, best_alpha=True).T
        
        n_times = X.shape[0]
        num_batches = n_times // batch_size
        
        scores = []
        for batch_id in range(num_batches):
            x_batch = X[batch_id*batch_size:(batch_id+1)*batch_size, :]
            p_batch = self.predict(x_batch, best_alpha=True).T
            # p_batch = predictions[batch_id*batch_size:(batch_id+1)*batch_size]
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            scores.append([pearsonr(p_batch[:, opc], y_batch[:, opc])[0] for opc in range(self.n_output_features)])
            
        return np.asarray(scores)
    
    def model_selection(self, X, y):

        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        returns:
        - mean_scores, ndarray of shape (n_alphas,)
        
        also sets the attribute best_alpha_idx.
        '''
        
        scores = self.score(X, y, best_alpha=False)
        mean_scores = np.mean(scores, axis=1)
        self.best_alpha_idx = np.argmax(mean_scores)
        return mean_scores
        
    # New lagged function: Introduce a matrix of shape (C, T) and return a matrix (L*C, T)
    def _get_lagged_matrix(self, X):
        n_chan, n_times = X.shape
        lagged_matrix = np.zeros((n_chan * self.num_lags, n_times))
        if self.start_lag < 0:
            range = np.arange(self.end_lag, self.start_lag, -1)
        else:
            range = np.arange(self.start_lag, self.end_lag)

        for i, lag in enumerate(range):
            
            shifted_X = np.roll(X, shift=lag, axis=1)

            # Rellenamos los elementos desplazados con ceros según el signo de lag
            if lag > 0:
                shifted_X[:, :lag] = 0  # Zerofill beginning if lag > 0
            elif lag < 0:
                shifted_X[:, lag:] = 0  # Zerofill end if lag < 0

            # Insertamos el canal desplazado en su posición en la matriz lageada
            lagged_matrix[i * n_chan:(i + 1) * n_chan, :] = shifted_X

        return lagged_matrix
    
class Ridge_SKL:

    def __init__(self, start_lag=0, end_lag=50, alpha=1, trial_len=3200, per_trial:bool = False):

        '''
        alpha: the regularisation parameter(s).
        trial_len: samples for each trial of the ds
        '''
        self.start_lag=start_lag
        self.end_lag = end_lag
        self.num_lags = self.end_lag-self.start_lag
        self.trial_len = trial_len
        if self.end_lag>0 and self.start_lag<0:
            self.num_lags+=1
        self.per_trial = per_trial
        self.best_alpha_idx=False
        
        if isinstance(alpha, Iterable):
            self.alphas = alpha
        else:
            self.alphas = np.array([alpha])
            self.best_alpha_idx=0

    
    def train(self, X_train, y_train, X_val, y_val):
        
        '''

        Get the best model from all the possible alphas

        inputs:
        - X, ndarray of shape (n_times, n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        
        '''
        
        # 1. Check that data shapes make sense
        print("Checking inputs...")
        n_times, self.n_input_features = X_train.shape
        n_output_times, self.n_output_features = y_train.shape
        n_times_val = X_val.shape[0]
        assert n_times==n_output_times
        
        # 2. Compute the lagged matrices
        if self.per_trial:
            n_trials = n_times // self.trial_len
            lagged_matrix = np.zeros((self.num_lags*self.n_input_features, n_times))
            lagged_matrix_val = np.zeros((self.num_lags*self.n_input_features, n_times_val))
            for n in tqdm(range(n_trials), desc="Computing lagged matrix"):
                start = n * self.trial_len
                end = start + self.trial_len
                lagged_matrix[:, start:end] = self._get_lagged_matrix(X_train[start:end, :].T)
                lagged_matrix_val[:, start:end] = self._get_lagged_matrix(X_val[start:end, :].T)
        else:
            lagged_matrix = self._get_lagged_matrix(X_train.T)
            lagged_matrix_val = self._get_lagged_matrix(X_val.T)
        lagged_matrix, lagged_matrix_val = lagged_matrix.T, lagged_matrix_val.T # (L*C, T) => (T, L*C)
        
        # 3. Perform Ridge
        list_scores = np.zeros((len(self.alphas), self.n_output_features))
        for i, alpha in tqdm(enumerate(self.alphas), desc='Iterating through alphas'):
            for j in range(self.n_output_features):
                
                # Fit the model with the train data
                ridge = Ridge_sklean(alpha)
                ridge.fit(lagged_matrix, y_train)
                
                # Select the best alpha with the validation data and add a new axis
                preds = ridge.predict(lagged_matrix_val)
                preds = preds.reshape(-1, 1)

                score = pearsonr(np.array(y_val[:, j], dtype=np.float64), preds[:, j])[0]
                list_scores[i, j] = score

        # Compute the mean between the output features for selecting the best alpha
        list_scores = np.mean(list_scores, axis=1)
        self.best_alpha = self.alphas[np.argmax(list_scores)]
        self.best_corr = np.max(list_scores)
        best_model = Ridge_sklean(alpha = self.best_alpha)
        best_model = best_model.fit(lagged_matrix, y_train)
        self.best_mdl = best_model

    def predict(self, X):

        '''
        
        Inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        Returns:
        - preds, ndarray of shape (n_times, n_output_features)
        
        '''

        n_times, n_input_features = X.shape
        assert n_input_features == self.n_input_features, f'Input must have {self.n_input_features} channels'

        # Compute the lagged matrix
        lagged_matrix = self._get_lagged_matrix(X.T)
        
        lagged_matrix = lagged_matrix.T # (L*C, T) => (T, L*C)

        preds = self.best_mdl.predict(lagged_matrix)

        return preds

    def score_in_batches(self, X, y, batch_size=125):
        
        '''
        
        Inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        Returns:
        - scores, ndarray of shape (n_output_features)
        
        '''
                
        n_times = X.shape[0]
        num_batches = n_times // batch_size
        
        scores = []
        
        for n in range(num_batches):

            # Get the batch data
            x_batch = X[n*batch_size:(n+1)*batch_size, :]
            y_batch = y[n*batch_size:(n+1)*batch_size, :]

            # Compute predictions
            p_batch = self.predict(x_batch)
            p_batch = p_batch.reshape(-1, 1)

            # Get the scores (for loop when more than one output feature)
            scores.append([pearsonr(p_batch[:, opc], np.array(y_batch[:, opc], dtype=np.float64))[0] for opc in range(self.n_output_features)])
            
        return np.asarray(scores)

    # New lagged function: Introduce a matrix of shape (C, T) and return a matrix (L*C, T)
    def _get_lagged_matrix(self, X):
        n_chan, n_times = X.shape
        lagged_matrix = np.zeros((n_chan * self.num_lags, n_times))
        if self.start_lag < 0:
            range = np.arange(self.end_lag, self.start_lag, -1)
        else:
            range = np.arange(self.start_lag, self.end_lag)

        for i, lag in enumerate(range):
            
            shifted_X = np.roll(X, shift=lag, axis=1)

            # Rellenamos los elementos desplazados con ceros según el signo de lag
            if lag > 0:
                shifted_X[:, :lag] = 0  # Zerofill beginning if lag > 0
            elif lag < 0:
                shifted_X[:, lag:] = 0  # Zerofill end if lag < 0

            # Insertamos el canal desplazado en su posición en la matriz lageada
            lagged_matrix[i * n_chan:(i + 1) * n_chan, :] = shifted_X

        return lagged_matrix