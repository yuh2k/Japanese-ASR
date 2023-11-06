from dtw import *
import librosa
import numpy as np
from pydub import AudioSegment
from matplotlib import pyplot as plt

signal, sr = librosa.load("query.wav")
dur = len(signal) / 16000 *1000

feats_1 = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).transpose()
query = (feats_1 - np.mean(feats_1, axis=0)) / np.std(feats_1, axis=0)



signal, sr = librosa.load("utterance.wav")
ext = AudioSegment.from_wav("utterance.wav")
feats_2 = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).transpose()
utterance = (feats_2 - np.mean(feats_2, axis=0)) / np.std(feats_2, axis=0)

print(len(signal))
print(feats_2.shape)






i_start = 0
n_step=3
n_query = query.shape[0]
n_search = utterance.shape[0]
sweep_costs = []

while i_start <= n_search - n_query or i_start == 0:
    score = dtw(query, utterance[i_start:i_start + n_query], dist_method="cosine", keep_internals=True).distance
    print(score)
    sweep_costs.append(score)
    i_start += n_step


time = sweep_costs.index(np.min(sweep_costs))*3*10

ext = ext[time: time+dur]
ext.export("out.wav", format="wav")

plt.plot(sweep_costs)
plt.show()
