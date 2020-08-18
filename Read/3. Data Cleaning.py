import pandas as pd

#Cleaning and exporting the Numeric Features CSV File
rating = pd.read_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Raw Data Frames/Numeric Features.csv')

# Fixing the file names and labels
rating['files'] = rating['files'].map(lambda x: x[11:-2])

# Mapping the labels to numeric values
label_map = {
    'ok': 1,
    'good': 2,
    'great': 3,
}

rating['y'] = rating['labels'].map(label_map)

rating.to_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Cleaned Data Frames/Numeric Features Cleaned.csv')

#Cleaning and exporting the Mel Spectrogram CSV File

mel_specs = pd.read_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Raw Data Frames/Mel Spectrogram Data.csv')

mel_specs = mel_specs.rename(columns={'84480': 'labels'})
mel_specs['y'] = mel_specs['labels'].map(label_map)

mel_specs.to_csv(r'D:/Python/Projects/Music Recommender/CSV Files/Cleaned Data Frames/Mel Spectrogram Cleaned.csv')