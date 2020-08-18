import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import math

start_time = time.time()

def extract_audio_features(directory):
    '''
    This function takes in a directory of .wav files and returns a
    DataFrame that includes several numeric features of the audio file
    as well as the corresponding genre labels.

    The numeric features incuded are the first 13 mfccs, zero-crossing rate,
    spectral centroid, and spectral rolloff.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    df (DataFrame): a table of audio files that includes several numeric features
    and genre labels.
    '''

    num = 1

    # Creating an empty list to store all file names
    files = []
    labels = []
    zcrs = []
    spec_centroids = []
    spec_rolloffs = []
    mfccs_1 = []
    mfccs_2 = []
    mfccs_3 = []
    mfccs_4 = []
    mfccs_5 = []
    mfccs_6 = []
    mfccs_7 = []
    mfccs_8 = []
    mfccs_9 = []
    mfccs_10 = []
    mfccs_11 = []
    mfccs_12 = []
    mfccs_13 = []

    # Looping through each file in the directory
    for file in os.scandir(directory):
        # Loading in the audio file
        y, sr = librosa.core.load(file)

        # Adding the file to our list of files
        files.append(file)

        # Adding the label to our list of labels
        label = str(file).split('.')[0]

        if "ok" in label:
            labels.append("ok")
        elif "good" in label:
            labels.append("good")
        else:
            labels.append("great")

        # Calculating zero-crossing rates
        zcr = librosa.feature.zero_crossing_rate(y)
        zcrs.append(np.mean(zcr))

        # Calculating the spectral centroids
        spec_centroid = librosa.feature.spectral_centroid(y)
        spec_centroids.append(np.mean(spec_centroid))

        # Calculating the spectral rolloffs
        spec_rolloff = librosa.feature.spectral_rolloff(y)
        spec_rolloffs.append(np.mean(spec_rolloff))

        # Calculating the first 13 mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfccs_1.append(mfcc_scaled[0])
        mfccs_2.append(mfcc_scaled[1])
        mfccs_3.append(mfcc_scaled[2])
        mfccs_4.append(mfcc_scaled[3])
        mfccs_5.append(mfcc_scaled[4])
        mfccs_6.append(mfcc_scaled[5])
        mfccs_7.append(mfcc_scaled[6])
        mfccs_8.append(mfcc_scaled[7])
        mfccs_9.append(mfcc_scaled[8])
        mfccs_10.append(mfcc_scaled[9])
        mfccs_11.append(mfcc_scaled[10])
        mfccs_12.append(mfcc_scaled[11])
        mfccs_13.append(mfcc_scaled[12])

        print("Finished Song: " + str(num))

        num += 1

    # Creating a data frame with the values we collected
    df = pd.DataFrame({
        'files': files,
        'zero_crossing_rate': zcrs,
        'spectral_centroid': spec_centroids,
        'spectral_rolloff': spec_rolloffs,
        'mfcc_1': mfccs_1,
        'mfcc_2': mfccs_2,
        'mfcc_3': mfccs_3,
        'mfcc_4': mfccs_4,
        'mfcc_5': mfccs_5,
        'mfcc_6': mfccs_6,
        'mfcc_7': mfccs_7,
        'mfcc_8': mfccs_8,
        'mfcc_9': mfccs_9,
        'mfcc_10': mfccs_10,
        'mfcc_11': mfccs_11,
        'mfcc_12': mfccs_12,
        'mfcc_13': mfccs_13,
        'labels': labels
    })

    return df


end_time = time.time()

total = end_time - start_time

minutes = math.floor(total / 60)
seconds = total % 60

df = extract_audio_features('D:/Python/Projects/Music Recommender/Database')

df.to_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Raw Data Frames/Numeric Features.csv', index=False)

print("Proccess Took: " + str(minutes) + " minutes and " + str(seconds) + " seconds")