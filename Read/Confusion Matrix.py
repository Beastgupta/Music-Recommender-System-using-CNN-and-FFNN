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
    1: 'ok',
    2: 'good',
    3: 'great',
}

# Renaming rows and columns with labes
confusion_df = confusion_df.rename(columns=labels_dict)
confusion_df.index = confusion_df.columns

# Creating a heatmap for the confusion matrix for display
plt.figure(figsize= (20,12))
sns.set(font_scale = 2)
ax = sns.heatmap(confusion_df, annot=True, cmap=sns.cubehelix_palette(50))
ax.set(xlabel='Predicted Values', ylabel='Actual Values')