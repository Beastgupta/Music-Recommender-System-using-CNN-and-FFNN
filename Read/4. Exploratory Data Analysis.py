# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display

df = pd.read_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Cleaned Data Frames/Numeric Features Cleaned.csv')


def plot_spectrogram(rating):
    '''
    This function takes in a list of genres and plots a mel spectrogram for one song
    per genre.
    '''

    # Loading in the audio file
    y, sr = librosa.core.load(f'D:/Python/Projects/Music Recommender/Database/{rating}. 1.wav')

    # Computing the spectrogram and transforming it to the decibal scale
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)  # Converting to decibels

    # Plotting the transformed spectrogram
    plt.figure(figsize=(10, 7))
    librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time')
    plt.title(str(rating))
    plt.show()

# Creating a list of all the genres
ratings = list(df['labels'].unique())

# Plotting spectrogram for each genre
"""for rating in ratings:
    plot_spectrogram(rating)
"""

def spectrogram_subplots(rating):
    '''
    This function takes in a list of genres and plots a mel spectrogram for one song
    per genre in a 5 x 2 grid.
    '''

    # Defining the subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax = ax.ravel()  # Turning ax into a matrix to make it easier to work with

    # Looping through the list of genres
    for i, kind in enumerate(rating):
        # Reading in the first file from each genre
        y, sr = librosa.core.load(f'D:/Python/Projects/Music Recommender/Database/{kind}. 1.wav')

        # Computing the mel spectrogram
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Displaying the mel spectrogram
        librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time', ax=ax[i])
        ax[i].set_title(str(kind))

    plt.show()

#spectrogram_subplots(ratings)


# Checking correlations
plt.figure(figsize=(10,7))
sns.heatmap(df.corr()[['y']].sort_values('y', ascending=False), annot=True)
plt.show()