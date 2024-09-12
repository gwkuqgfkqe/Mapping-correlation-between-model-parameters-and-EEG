# 导入包模块
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

with open('/weights/single_train/B2015_004_weights_g.txt', 'r') as f:

    lines = f.readlines()
    count=0

    for f in lines:
        if count==0:
            count=1
            aa = f
            my_list = aa.split()
            X = np.array(my_list)
            X = [float(X[i]) for i in range(30)]
            X = np.expand_dims(X, axis=0)
        else:
            aa=f
            my_list = aa.split()
            my_list=np.array(my_list)
            my_list = [float(my_list[i]) for i in range(30)]
            my_list = np.expand_dims(my_list, axis=0)
            X=np.concatenate((X, my_list), axis=0)

        print(X)



y=np.array([0]*8+[1]*8+[2]*8+[3]*8+[4]*8)




pca = PCA(n_components=2)
pca = pca.fit(X)
X_dr = pca.transform(X)


colors = ['red', 'black', 'orange','blue','green']
labels = ['feet', 'navigation', 'right_hand','subtraction','word_ass']

#fig = plt.figure()
plt.figure()
#ax = fig.add_subplot(111, projection='3d')
for i in [0, 1,2,3,4]:
    plt.scatter(X_dr[y == i, 0]
                , X_dr[y == i, 1]
                # ,X_dr[y == i, 2]

                , alpha=.7
                , c=colors[i]
                ,label=labels[i]
                # , label=iris_ds.target_names[i]
                )
plt.legend()
#plt.title('PCA of IRIS dataset')
plt.show()
