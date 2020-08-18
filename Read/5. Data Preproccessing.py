# Imports
import os
import librosa
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import utils
from keras.utils import to_categorical
import time
import math

start_time = time.time()

def extract_mel_spectrogram(directory):
    '''
    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for each audio file, reshapes them so that they are all the
    same size, flattens them, and stores them in a dataframe.

    Genre labels are also computed and added to the dataframe.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    df (DataFrame): a dataframe of flattened mel spectrograms and their
    corresponding genre labels
    '''

    num = 0

    # Creating empty lists for mel spectrograms and labels
    labels = []
    mel_specs = []

    # Looping through each file in the directory
    for file in os.scandir(directory):

        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Extracting the label and adding it to the list
        label = str(file).split('.')[0]
        labels.append(label)

        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adjusting the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128, 660, refcheck=False)

        # Adding the mel spectrogram to the list
        mel_specs.append(spect)

        num += 1

        print("Finished song number: " + str(num))

    # Converting the list or arrays to an array
    X = np.array(mel_specs)

    # Converting labels to numeric values
    labels = pd.Series(labels)
    label_dict = {
        '<DirEntry \'ok': int(1),
        '<DirEntry \'good': int(2),
        '<DirEntry \'great': int(3),
    }
    y = labels.map(label_dict).values

    # Returning the mel spectrograms and labels
    return X, y

X, y = extract_mel_spectrogram(r'D:\Python\Projects\Music Recommender\Database')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.2)

# Checking the minimum value (the scale ranges from zero to some negative value) to see how we should scale the data
X_train_min = X_train.min()

X_train /= math.floor(X_train_min)
X_test /= math.floor(X_train_min)

print(math.floor(X_train_min))

# Reshaping images to be 128 x 660 x 1, where the 1 represents the single color channel
X_train = X_train.reshape(X_train.shape[0], 128, 660, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 660, 1)


# One hot encoding our labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

end_time = time.time()

total = end_time - start_time

minutes = math.floor(total / 60)
seconds = total % 60

print("Process Took: " + str(minutes) + " minutes and " + str(seconds) + " seconds")