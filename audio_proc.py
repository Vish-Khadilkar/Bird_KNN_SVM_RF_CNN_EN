import os
from pydub import AudioSegment
import numpy as np
import pandas as pd
import librosa
import librosa.display
import config as cfg
import cv2
import matplotlib.pyplot as plt
import IPython.display as ipd
from PIL import Image

CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train')))]

for c in CLASSES:
    os.mkdir(os.path.join(cfg.TRAINSET_PATH, 'train_chunk', c))
    ifiles = [f for f in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train', c)))]
    for aud_read in ifiles:
        readpath = os.path.join(cfg.TRAINSET_PATH, 'train', c, aud_read)
        data2, sampling_rate2 = librosa.load(readpath, 44000)
        newAudio = AudioSegment.from_wav(readpath)
        dur = len(data2)/sampling_rate2
        # pydub lib
        i = 1
        while i < dur:
            if dur - i > 3:
                writepath = cfg.TRAINSET_PATH + '/' + 'train_chunk' + '/' + c + '/' + str(i) + aud_read
                chunk = newAudio[(i * 1000):((i + 2) * 1000)]
                chunk.export(writepath, format="wav")
                af, s = librosa.load(writepath)
                print(np.array(librosa.feature.melspectrogram(af)).shape)
                melspec = librosa.feature.melspectrogram(af, n_fft=128 * 4, n_mels=128, hop_length=len(af) // (256 - 1))
                # Convert power spec to dB scale (compute dB relative to peak power)
                #melspec = librosa.amplitude_to_db(melspec, ref=np.max, top_db=80)
                # Flip spectrum vertically (only for better visualization, low freq. at bottom)
                melspec = melspec[::-1, ...]
                # Trim to desired shape if too large
                melspec = melspec[:128, :256]
                cv2.imwrite(writepath + '.png', melspec)
            i = i + 3


