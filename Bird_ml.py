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
X.shape = (3974, -1) #for cut 3 species example
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

## Start Execution of Classification ##

####################
#SVM PARAMETER
print('Running SVM..')
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3],
                     'C': [0.001, 0.10]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3],
                     'C': [0.001, 0.10]},
                    {'kernel': ['linear'], 'C': [0.001]}]
clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2)
clf.fit(x_train_scaled_PCA, y_train)
y_predict2 = clf.predict(x_test_scaled_PCA)

#save best model for SVM
svm_best = clf.best_estimator_
#check best n_neigbors value
print('Best_params')
print(clf.best_params_)
#check  best model
print('model.best.estimator_')
print(svm_best)
print('Results for SVM')
print(f'Confusion Matrix: \n{confusion_matrix(y_predict2, y_test)}')
print(f'SVM Model Score: {clf.score(x_test_scaled_PCA, y_test)}')
####################


##KNN
grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled_PCA, y_train)
print('Results for KNN')
print(f'KNN Model Score: {model.score(x_test_scaled_PCA, y_test)}')

#save best model
knn_best = model.best_estimator_
#check best n_neigbors value
print('Best_params')
print(model.best_params_)
#check  best model
print('model.best.estimator_')
print(knn_best)

y_predict = model.predict(x_test_scaled_PCA)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')
#class_names = ['Baryphthengus ruficapillus', 'Hypocnemis cantator', 'Notiochelidon cyanoleuca']
# Plot normalized confusion matrix
#ncm.plot_confusion_matrix(y_test, y_predict, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix for KNN')

##SVM
###old copied code####
#parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
#svc = SVC(gamma="scale")
#clf = GridSearchCV(SVC, cv=5)
###end of old copied code####
for kernel in ('linear', 'rbf', 'poly'):
    clf = SVC(kernel=kernel, gamma=10)
    clf.fit(x_train_scaled_PCA, y_train)

#sorted(clf.cv_results_.keys())
    print('Results for SVM {kernel}')
    print(f'SVM Model Score for {kernel}: {clf.score(x_test_scaled_PCA, y_test)}')
    y_predict2 = clf.predict(x_test_scaled_PCA)
    print(f'Confusion Matrix: \n{confusion_matrix(y_predict2, y_test)}')
#plt.show()

####################

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

##Naive Bayes##
model_nb = GaussianNB()
model_nb.fit(x_train_scaled_PCA,y_train)
results_nb = model_nb.predict(x_test_scaled_PCA)
#print(results_nb.mean())
#print(results_nb)
print('Results for Naive Bayes')
print(f'Confusion Matrix: \n{confusion_matrix(results_nb, y_test)}')

##Ensemble##
estimators = []
estimators.append(('RandomForestClassifier', clf_rf))
estimators.append(('knn', model))
estimators.append(('svm', clf))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, x_train_scaled_PCA, y_train, cv=10)
print('Results for Ensemble')
print(results.mean())
print(results)
#print(f'Confusion Matrix: \n{confusion_matrix(results, y_test)}')

##Adaboost##
seed = 7
num_trees = 50
kfold = model_selection.KFold(n_splits=20, random_state=seed)
model_adaboost = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results_adaboost = model_selection.cross_val_score(model_adaboost, x_train_scaled_PCA, y_train, cv=kfold)
print('Results for Adaboost')
print(results_adaboost.mean())
print(results_adaboost)
#print(f'Confusion Matrix: \n{confusion_matrix(results_adaboost, y_test)}')


