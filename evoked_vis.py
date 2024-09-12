from braindecode.datasets import MOABBDataset
import matplotlib.pyplot as plt
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2015_004", subject_ids=[subject_id],)

import numpy as np
import mne
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

transforms = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    # Preprocessor(
    #     lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
    #     factor=1e6,
    # ),
    # Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, transforms, n_jobs=-1)

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)




splitted = windows_dataset.split("session")
train_set = splitted['0']  # Session train
test_set = splitted['1']  # Session evaluation
AA=0
flag=0
epochs_data=0
montage = mne.channels.make_standard_montage("standard_1005")
for i in range(200):
    if train_set[i][1]==4:
        if flag==0:
            epochs_data=test_set[i][0][np.newaxis,:,:]
            flag=1
        else:
            middle=test_set[i][0][np.newaxis,:,:]
            epochs_data=np.concatenate((epochs_data,middle),axis=0)
evoked_data = np.mean(epochs_data, axis=0)
info=test_set.datasets[0].raw.info
nave = len(epochs_data)
evokeds = mne.EvokedArray(evoked_data, info=info, tmin=-0.5,comment='Arbitrary', nave=nave)
evokeds.set_montage(montage)
#evokeds.plot( show=True, time_unit='s',spatial_colors=True)
# plt.show()
ts_args = dict( time_unit='s')
topomap_args = dict(sensors=False, time_unit='s')
evokeds.plot_joint( times='peaks',ts_args=ts_args, topomap_args=topomap_args)
evokeds.plot_image(exclude=[], time_unit='s')
evokeds.plot_topomap(times='peaks', time_unit='s')#,show_names=True)
plt.show()