# Music-Recommender-System-using-CNN-and-FFNN
Trains a CNN or FFNN on a database of spectrogram of music files specified by the user in order to predict three labels for the user: "ok", "good", and "great".

Used work by Leland Roberts from his article: Musical Genre Classification with Convolutional Neural Networks. I adapted the code and changed it for use for my purpose.

Article: https://towardsdatascience.com/musical-genre-classification-with-convolutional-neural-networks-ff04f9601a74
GitHub Repo: https://github.com/lelandroberts97/Musical_Genre_Classification

In order to run this program the user needs to follow these steps:

1) Download songs that they like into a .WAV format

2) Rename each song by this format - {"ok", "good", or "great" based on the users' preference of the song}. {file number for specified song name} 

(Example) 

[![dO00nn.md.png](https://iili.io/dO00nn.md.png)](https://freeimage.host/i/dO00nn)

3) Open all the Python files in the Read folder and change the directories to suit wherever you are downloading and creating your files on your computer system.

4) Run the files in order from 1-7.

NOTE:

1) The 5 files with no number were created for framework purposes and testing before being used in the main python files.

2) The second Python Files titled "2. Read and Extract Numeric Features from Audio Files and Export Dataframe as CSV" is not neccessary to run as it creates Numeric Feature data which is currently not being used to train a model on.
