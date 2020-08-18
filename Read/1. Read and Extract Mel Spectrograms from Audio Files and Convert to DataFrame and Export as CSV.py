import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import math

start_time =  time.time()


def make_mel_spectrogram_df(directory):
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
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)

        # Adjusting the size to be 128 x 660
        if spect.shape[1] != 660:
            spect.resize(128, 660, refcheck=False)

        # Flattening to fit into dataframe and adding to the list
        spect = spect.flatten()
        mel_specs.append(spect)

        num += 1

        print("Finished song number: " + str(num))

    # Getting number of files in directory to pass them into array to reshape
    mp3_files_in_directory = next(os.walk(directory))[2]
    num_of_files = len(mp3_files_in_directory)

    # Converting the lists to arrays so we can stack them
    mel_specs = np.array(mel_specs)
    labels = np.array(labels).reshape(num_of_files, 1)

    # Create dataframe
    df = pd.DataFrame(np.hstack((mel_specs, labels)))

    # Returning the mel spectrograms and labels
    return df


df = make_mel_spectrogram_df('D:/Python/Projects/Music Recommender/Database')


end_time = time.time()

total = end_time - start_time

minutes = math.floor(total / 60)
seconds = total % 60

df.to_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Raw Data Frames/Mel Spectrogram Data.csv', index=False)

print("Proccess Took: " + str(minutes) + " minutes and " + str(seconds) + " seconds")

