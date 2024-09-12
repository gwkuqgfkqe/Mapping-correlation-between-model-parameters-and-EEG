

import numpy as np


lists=list(range(41))
with open('/weights/single_train/B2015_004_weights_g.txt', 'r') as f:

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
                X = [float(X[i]) for i in range(30)]
                X = np.expand_dims(X, axis=0)
                #counts+=1
        else:
            if counts in lists:
                aa=f

                my_list = aa.split()
                my_list=np.array(my_list)
                my_list = [float(my_list[i]) for i in range(30)]
                my_list = np.expand_dims(my_list, axis=0)
                X=np.concatenate((X, my_list), axis=0)
                #counts+=1
        counts+=1
        print(X)



#a=np.concatenate((np.expand_dims(X[:,:],axis=0),np.expand_dims(X[100:,:],axis=0)),axis=0)#,np.expand_dims(X[16:24,:],axis=0),np.expand_dims(X[24:,:],axis=0)),axis=0)#,np.expand_dims(X[32:,:],axis=0)),axis=0)
a=np.sum(X[32:], axis=0)


import matplotlib.pyplot as plt


for i in range(len(my_list)):
    my_list[i]=float(my_list[i])
data = my_list


plt.plot(range(len(data)), data)
plt.title('1x500 List Plot')
plt.xlabel('Frequency')
plt.ylabel('Weights')

# 显示图形
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# 创建一个随机矩阵
with open('/weights/single_train/B2015_001_weights_f.txt', 'r') as f:
    matrix = np.random.rand(1409, 1409)
    i=0
    lines = f.readlines()
    for line in lines:
        # 处理每行数据

        a=line.split()
        for j in range(1409):
            matrix[i][j]=float(a[j])
        i+=1

plt.imshow(matrix[:100,:50], cmap='coolwarm', interpolation='nearest')

plt.colorbar()

plt.title('Matrix Visualization')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.show()
# plt.savefig('frequency_matrix.png')
from braindecode.datasets import MOABBDataset
import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
dataset=MOABBDataset(dataset_name="BNCI2015_004", subject_ids=[3])

raw=dataset.datasets[0].raw

#montage = mne.channels.make_standard_montage("standard_1020")
info = raw.info
a=a.astype(float)
a=a.reshape((30,))
#np.squeeze(arr)
#fig = plt.figure(figsize=(10, 5))
#plt.figure(figsize=(10, 5))
# gs = GridSpec(1, 2, width_ratios=[9, 1])
#

# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# fig.subplots_adjust( wspace=0.1)
# mne.viz.plot_brain_colorbar(ax2,'auto',colormap= 'coolwarm')
mne.viz.plot_topomap(
    a, pos=info,names=info.ch_names,vlim=(a.min(),a.max()),cmap= 'RdBu'
)
#axes=ax1,
#plt.colorbar()
plt.show()

