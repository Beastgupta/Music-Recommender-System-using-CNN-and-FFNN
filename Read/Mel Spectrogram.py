import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

#Importing Specific .WAV File
y, sr = librosa.load('C:/Users/User/Downloads/Skan - Run (1).wav')


# Computing the mel spectrogram
spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
spect = librosa.power_to_db(spect, ref=np.max) # Converting to decibals

# Extracting mfccs from the audio signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)

# Displaying the mfccs
plt.figure(figsize=(8,5))
librosa.display.specshow(mfcc, x_axis='time')
plt.title('MFCC')

#Displaying the Mel Spectrogram
plt.figure(figsize=(8,5))
librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')



#Displaying Graphs
plt.show()