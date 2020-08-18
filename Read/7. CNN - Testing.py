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
import tensorflow as tf

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
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256)
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
        '<DirEntry \'ok': int(0),
        '<DirEntry \'good': int(1),
        '<DirEntry \'great': int(2),
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


# Reshaping images to be 128 x 660 x 1, where the 1 represents the single color channel
X_train = X_train.reshape(X_train.shape[0], 128, 660, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 660, 1)


# One hot encoding our labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

np.random.seed(23456)
tf.random.set_seed(123)

# Initiating an empty neural network
cnn_model = Sequential(name='cnn_1')

# Adding convolutional layer
cnn_model.add(Conv2D(filters=16,
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(128,660,1)))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding convolutional layer
cnn_model.add(Conv2D(filters=32,
                     kernel_size=(3,3),
                     activation='relu'))

# Adding max pooling layer
cnn_model.add(MaxPooling2D(pool_size=(2,4)))

# Adding a flattened layer to input our image data
cnn_model.add(Flatten())

# Adding a dense layer with 64 neurons
cnn_model.add(Dense(64, activation='relu'))

# Adding a dropout layer for regularization
cnn_model.add(Dropout(0.25))

# Adding an output layer
cnn_model.add(Dense(10, activation='softmax'))

# Compiling our neural network
cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# Fitting our neural network
history = cnn_model.fit(X_train,
                        y_train,
                        batch_size=16,
                        validation_data=(X_test, y_test),
                        epochs=15)



# Check out our train loss and test loss over epochs.
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(test_loss, label='Testing Loss', color='red')

# Set title
plt.title('Training and Testing Loss by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(1,16), range(1,16))

plt.legend(fontsize = 18)



# Check out our train accuracy and test accuracy over epochs.
train_loss = history.history['accuracy']
test_loss = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_loss, label='Training Accuracy', color='blue')
plt.plot(test_loss, label='Testing Accuracy', color='red')

# Set title
plt.title('Training and Testing Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Accuracy', fontsize = 18)
plt.xticks(range(1,21), range(1,21))

plt.legend(fontsize = 18);



# Making predictions from the cnn model
predictions = cnn_model.predict(X_test, verbose=1)



# Checking the number of targets per class
for i in range(10):
    print(f'{i}: {sum([1 for target in y_test if target[i] == 1])}')

# Checking the number of predicted values in each class
for i in range(10):
    print(f'{i}: {sum([1 for prediction in predictions if np.argmax(prediction) == i])}')

# Calculating the confusion matrix
# row: actual
# columns: predicted
conf_matrix = confusion_matrix(np.argmax(y_test, 1), np.argmax(predictions, 1))

# Creating a dataframe of the confusion matrix with labels for readability
confusion_df = pd.DataFrame(conf_matrix)


# Creating a dictionary of labels
labels_dict = {
    0: 'ok',
    1: 'good',
    2: 'great',
}

# Renaming rows and columns with labes
confusion_df = confusion_df.rename(columns=labels_dict)
confusion_df.index = confusion_df.columns

# Creating a heatmap for the confusion matrix for display
plt.figure(figsize= (20,12))
sns.set(font_scale = 2)
ax = sns.heatmap(confusion_df, annot=True, cmap=sns.cubehelix_palette(50))
ax.set(xlabel='Predicted Values', ylabel='Actual Values')

plt.show()

end_time = time.time()

total = end_time - start_time

minutes = math.floor(total / 60)
seconds = total % 60

print("Process Took: " + str(minutes) + " minutes and " + str(seconds) + " seconds")