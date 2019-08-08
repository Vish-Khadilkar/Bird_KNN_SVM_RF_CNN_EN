import norm_conf_mat as ncm

import os

import numpy as np
import pandas as pd

import librosa
import librosa.display
import soundfile as sf # librosa fails when reading files on Kaggle.

import matplotlib.pyplot as plt
import IPython.display as ipd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

import config as cfg

np_aud = []
samples = []
Ya = []

CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'spec')))]

for c in CLASSES:
    filepath = os.path.join(cfg.DATASET_PATH, c)
    print('Processing Audio Recordings For Species: ' + c)
    for file in os.listdir(filepath):
        if file.endswith(".txt"):
            #print(os.path.join(filepath, file))
            np_aud = np.loadtxt(os.path.join(filepath, file), delimiter=',', unpack=True)
            #print(np.shape(np_aud))
            Ya.append(c)
            samples.append(np_aud)
    #X = np.append(X, np.array(samples), axis=0)
    #Y = np.append(Y, np.array(Ya), axis=0)
    #print(X[0])
    #print(X.shape)
    #print(Y.shape)
#print(np.shape(np.array(samples)))
#print(np.shape(np.array(Ya)))

X = np.array(samples)
#X = X.flatten('F')[:X.shape[0]]
X = X.flatten('F')
#X.shape = (1472, -1) #for 2 species example
#X.shape = (4928, -1) #for 10 species example?
#X.shape = (1940, -1) #for 3 species example
X.shape = (1454, -1) #for cut 3 species example
#X.shape = (2570, -1) #for 4 species example
#X.shape = (-1, 32768) #for all? species example
##not used#######X = X.transpose(2, 0, 1).reshape(1472, -1)
Y = np.array(Ya)

#print(np.shape(X))
#print(np.shape(Ya))

x_train, x_test, y_train, y_test = train_test_split(X, Y)
#print(f'Shape: {x_train.shape}')
#print(f'Observation: \n{x_train[0]}')
#print(f'Labels: {y_train[:10]}')

#print(f'Shape: {x_test.shape}')
#print(f'Observation: \n{x_test[0]}')
#print(f'Labels: {y_test[:10]}')

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA().fit(x_train_scaled)

#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)')
#plt.show()

x_train_scaled_PCA = pca.transform(x_train_scaled)
x_test_scaled_PCA = pca.transform(x_test_scaled)
#print(x_train_scaled_PCA.shape)
#print(x_test_scaled_PCA.shape)


##Random Forest

clf_rf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=20)

clf_rf.fit(x_train_scaled_PCA, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

preds = clf_rf.predict(x_test_scaled_PCA)

#pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result'])
print('Results for Random Forest')
#print(f'Random Forest Model Score: {preds.score(x_test_scaled_PCA, y_test)}')

print(f'Confusion Matrix: \n{confusion_matrix(preds, y_test)}')

#plt.show()
