import sys
import os 
SCRIPT_DIR = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(SCRIPT_DIR))

import mne
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import mlab as mlab
from utils.datasets import CustomDataset
from utils.functional import get_data_path, get_trials_len

data_path = get_data_path('c:/users/jaulab/desktop/AAD/Data', 'fulsang', filt=False)
# from preprocess import import_data
dataset = CustomDataset('fulsang', data_path, 'train', 'S1', window=50, hop=1)
trial_len = get_trials_len('fulsang')
chan, samples = dataset.eeg.size()
data = dataset.eeg.view(trial_len, chan, samples // trial_len)
label = 0
# get the data and label (label that indicates the trials)
# data - (samples, channels, trials)
# label -  (label, 1)

data = np.transpose(data, (2, 1, 0))
data = np.array(data, dtype=np.float64)
# label = np.squeeze(np.transpose(label))
# idx = np.where(label == 1)
# data_draw = data[idx]

mean_trial = np.mean(data, axis=0)  # mean trial
# use standardization or normalization to adjust
mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)

mean_ch = np.mean(mean_trial, axis=1)  # mean samples with channel dimension left

# Draw topography
chan_idx = dataset.chan_idx
biosemi_montage = mne.channels.make_standard_montage('biosemi64')  # set a montage, see mne document
# montaje del egg => NO TOCAR!! Los montajes son propios del sistema y modificarlo spuede dar error
# index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]  # correspond channel
# chan_to_idx = {name: idx for idx, name in enumerate(biosemi_montage.ch_names)}
# index = [chan_to_idx[chan] for chan in chan_idx]
# biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
# biosemi_montage.dig = [biosemi_montage.dig[i] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=64., ch_types='eeg')  # sample rate

evoked1 = mne.EvokedArray(mean_trial, info)
evoked1.set_montage(biosemi_montage)
plt.figure(1)
# im, cn = mne.viz.plot_topomap(np.mean(mean_trial, axis=1), evoked1.info, show=False)
im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False)
plt.colorbar(im)

# plt.savefig('./topo/test.png')
plt.show()
print('the end')