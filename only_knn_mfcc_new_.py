import norm_conf_mat as ncm
import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import cv2

import config as cfg

np_aud = []
samples = []
Ya = []

CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train_chunk')))]


for c in CLASSES:
    ifiles = [f for f in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train_chunk', c)))]
    for f_read in ifiles:
        if f_read.endswith(".wav"):
            readpath = os.path.join(cfg.TRAINSET_PATH, 'train_chunk', c, f_read)
            data2, sampling_rate2 = librosa.load(readpath, 44000)
            mfccs = librosa.feature.mfcc(data2, sr=sampling_rate2)
            #print(mfccs)
            mfcc2 = np.roll(np.array(mfccs), 5)
            mfcc3 = np.roll(np.array(mfccs), 10)
            #print(f'MFFCs shape: {mfccs.shape}')
            #print(f'First mffcs: {mfccs[0, :5]}')
            np_aud = mfccs
            np_aud2 = mfcc2
            np_aud3 = mfcc3
            Ya.append(c)
            Ya.append(c)
            Ya.append(c)
            samples.append(np_aud)
            samples.append(np_aud2)
            samples.append(np_aud3)

X = np.array(samples)
print(X.shape)
X = X.flatten('F')
print(X.shape)
X.shape = (3440, -1)
print(X.shape)
X = np.transpose(X)
print(X.shape)

Y = np.array(Ya)
print(np.shape(Ya))

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
    #'n_neighbors': [3, 5, 7, 9, 11, 15],
    'n_neighbors': [3, 5, 7, 9, 15, 20],
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

#y_predict = model.predict(x_test_scaled_PCA)
y_predict = knn_best.predict(x_test_scaled_PCA)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')
class_names = ['Cistothorus apolinari','Coereba flaveola','Cyphorhinus thoracicus','Rupornis magnirostris']
# Plot normalized confusion matrix
ncm.plot_confusion_matrix(y_test, y_predict, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
