import neurora.rdm_cal as rc
import neurora.rsa_plot
import neurora.rdm_corr
import neurora.corr_cal
import torch
import torchvision as tv
import numpy as np

# bhv_data1 = np.random.randn(5, 5, 4)  # the number of conidtions, the number of subjects, the number of trials
# bhv_data2 = np.random.randn(5, 5, 4)  # the number of conidtions, the number of subjects, the number of trials
# rdm1 = rc.bhvRDM(bhv_data1, sub_opt=0, method='correlation', abs=False)
# rdm2 = rc.bhvRDM(bhv_data2, sub_opt=0, method='correlation', abs=False)
#
# # 使用pearson，当然也可以用spearman等，如rdm_correlation_spearman
# spearman_result = neurora.rdm_corr.rdm_correlation_pearson(rdm1, rdm2, rescale=False, permutation=False, iter=5000)
# print(spearman_result)

#one=[1,2,3,4,5]
one=range(1,73,1)
two=range(73,145,1)
three=range(145,217,1)
four=range(217,289,1)
# two=[73,74,75,76,77]
# three=[145,146,147,148,149]2014_001_gcnn_combine.png
# four=[217,218,219,220,221]2014_001_gcnn.png
lists=list(one)+list(two)+list(three)+list(four)
lists=list(range(201))
with open('/weights/single_train/B2015_001_weights_f.txt', 'r') as f:
    # 写入一些内容到文件
    lines = f.readlines()
    count=0
    counts=1

    for f in lines:
        if count==0:
            if counts in lists:
                count=1
                aa = f

                my_list = aa.split()
                X = np.array(my_list)
                X = [float(X[i]) for i in range(1409)]
                X = np.expand_dims(X, axis=0)
                #counts+=1
        else:
            if counts in lists:
                aa=f

                my_list = aa.split()
                my_list=np.array(my_list)
                my_list = [float(my_list[i]) for i in range(1409)]
                my_list = np.expand_dims(my_list, axis=0)
                X=np.concatenate((X, my_list), axis=0)
                #counts+=1
        counts+=1
        print(X)



bhv_data1=np.concatenate((np.expand_dims(X[:100,:],axis=0),np.expand_dims(X[100:,:],axis=0)),axis=0)#,np.expand_dims(X[16:24,:],axis=0),np.expand_dims(X[24:,:],axis=0)),axis=0)#,np.expand_dims(X[32:,:],axis=0)),axis=0)
rdm1 = rc.bhvRDM(bhv_data1, sub_opt=0, method='correlation', abs=False)




from braindecode.datasets import MOABBDataset
import matplotlib.pyplot as plt
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2015_001", subject_ids=[subject_id],)

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
train_set = splitted['0A']  # Session train
test_set = splitted['1B']  # Session evaluation
AA=0
flag=0
flag2=0
flag3=0
flag4=0
flag5=0
epochs_data1=0
epochs_data2=0
epochs_data3=0
epochs_data4=0
epochs_data5=0
montage = mne.channels.make_standard_montage("standard_1005")
for i in range(200):
    print(i)
    if test_set[i][1]==0:
        if flag==0:
            epochs_data1=test_set[i][0][np.newaxis,:,:]
            flag=1
        else:
            if epochs_data1.shape[0]==100:
                continue
            middle=test_set[i][0][np.newaxis,:,:]
            epochs_data1=np.concatenate((epochs_data1,middle),axis=0)
    elif test_set[i][1]==1:
        if flag2==0:
            epochs_data2=test_set[i][0][np.newaxis,:,:]
            flag2=1
        else:
            if epochs_data2.shape[0]==100:
                continue
            middle=test_set[i][0][np.newaxis,:,:]
            epochs_data2=np.concatenate((epochs_data2,middle),axis=0)
    elif test_set[i][1]==2:
        if flag3==0:
            epochs_data3=test_set[i][0][np.newaxis,:,:]
            flag3=1
        else:
            if epochs_data3.shape[0]==72:
                continue
            middle=test_set[i][0][np.newaxis,:,:]
            epochs_data3=np.concatenate((epochs_data3,middle),axis=0)
    elif test_set[i][1]==3:
        if flag4==0:
            epochs_data4=test_set[i][0][np.newaxis,:,:]
            flag4=1
        else:
            if epochs_data4.shape[0]==72:
                continue
            middle=test_set[i][0][np.newaxis,:,:]
            epochs_data4=np.concatenate((epochs_data4,middle),axis=0)
    else:
        if flag5==0:
            epochs_data5=test_set[i][0][np.newaxis,:,:]
            flag5=1
        else:
            if epochs_data5.shape[0]==40:
                continue
            middle=test_set[i][0][np.newaxis,:,:]
            epochs_data5=np.concatenate((epochs_data5,middle),axis=0)
epochs_data1=np.expand_dims(np.expand_dims(epochs_data1,axis=0),axis=0)
epochs_data2=np.expand_dims(np.expand_dims(epochs_data2,axis=0),axis=0)
epochs_data3=np.expand_dims(np.expand_dims(epochs_data3,axis=0),axis=0)
epochs_data4=np.expand_dims(np.expand_dims(epochs_data4,axis=0),axis=0)
epochs_data5=np.expand_dims(np.expand_dims(epochs_data5,axis=0),axis=0)
epochs_datax=np.concatenate((epochs_data1,epochs_data2),axis=0)#,epochs_data3,epochs_data4),axis=0)
rdm2 = rc.eegRDM(epochs_datax, sub_opt=0, method='correlation', abs=False)

# rdm2 = rc.eegRDM(epochs_datay, sub_opt=0, method='correlation', abs=False)
#print(rdm1.shape)

plt.imshow(rdm1, cmap='rainbow', interpolation='nearest')
plt.imshow(rdm2, cmap='rainbow', interpolation='nearest')
# 显示颜色条


# plt.axis("off")
cb = plt.colorbar()
cb.ax.tick_params(labelsize=16)
font = {'size': 18}
cb.set_label("Dissimilarity", fontdict=font)
plt.show()
neurora.rsa_plot.plot_rdm_withvalue(rdm1, lim=[0, 1], value_fontsize=10, conditions=None, con_fontsize=12,
                                    cmap='rainbow')
neurora.rsa_plot.plot_rdm_withvalue(rdm2, lim=[0, 1], value_fontsize=10, conditions=None, con_fontsize=12,
                                    cmap='rainbow')