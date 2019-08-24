import norm_conf_mat as ncm
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

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
            np_aud = np.loadtxt(os.path.join(filepath, file), delimiter=',', unpack=True)
            Ya.append(c)
            samples.append(np_aud)

X = np.array(samples)
X = X.flatten('F')
#X.shape = (np.shape(np.array(Ya))[0], -1)
X.shape = (32768, -1)
X = np.transpose(X)
Y = np.array(Ya)

x_train, x_test, y_train, y_test = train_test_split(X, Y)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA().fit(x_train_scaled)

x_train_scaled_PCA = pca.transform(x_train_scaled)
x_test_scaled_PCA = pca.transform(x_test_scaled)

## Start Execution of Classification ##

##KNN
grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=10, n_jobs=-1)
model.fit(x_train_scaled_PCA, y_train)
print('Results for KNN')
print(f'KNN Model Score: {model.score(x_test_scaled_PCA, y_test)}')

#save best model
knn_best = model.best_estimator_
#check  best model
print('model.best.estimator_')
print(knn_best)
#check best n_neigbors value
print('Best_params')
print(model.best_params_)

y_predict = model.predict(x_test_scaled_PCA)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')
class_names = ['Cistothorus apolinari','Coereba flaveola','Cyphorhinus thoracicus','Rupornis magnirostris']
# Plot normalized confusion matrix
ncm.plot_confusion_matrix(y_test, y_predict, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for KNN')
