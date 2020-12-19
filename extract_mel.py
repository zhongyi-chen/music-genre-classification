import os
import sys
import pickle
import keras
import librosa
import numpy as np
import pandas as pd


class MelSpectrogram(object):
    def __init__(self, file_path, csv_file):
        # Constants
        self.song_samples = 639450
        self.n_fft = 2048
        self.hop_length = 2048  # 512 !
        self.tol = 10e-3
        self.file_path = file_path
        self.csv_file = csv_file
        self.genres = {'Classical': 1, 'Electronic': 2, 'Folk': 3, 'Hip-Hop': 4, 'WorldMusic': 5,
                       'EXperimental': 6, 'Pop': 7, 'Rock': 8}

    def getdata(self):
        # Array of songs and array of genres
        song_data = []
        genre_data = []

        genre = {}
        with open(self.csv_file) as f:
            train = map(lambda line: line.rstrip().split(','), f.readlines())
            t = list(train)
            t = t[1:]
            train = [(x[0], x[1]) for x in t]
            for t in train:
                genre[t[0]] = t[1]

        # Read files from the folders
        cpt = 0
        for t in train:
            # Read the audio file
            file_name = self.file_path + "/" + str(t[0]) + ".mp3"
            signal, sr = librosa.load(file_name, duration=29)
            # print(signal.shape)
            # print(sr)
            cpt += 1
            print(cpt, file_name)

            # Calculate the melspectrogram of the audio
            melspec = librosa.feature.melspectrogram(signal[:self.song_samples],
                                                     sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=128)

            # Append the result to the data structure
            # print(melspec.shape)
            song_data.append(melspec.T)
            genre_data.append(int(t[1])-1)

        return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))

    def normalize(self, songs):
        # Allocate memory
        norm_songs = np.zeros(songs.shape)
        for i in range(songs.shape[0]):
            # Subtrac the mean
            song_mean_channel = list(
                map(lambda x, y: x - y, songs[i], np.mean(songs[i], axis=1)))
            song_mean_channel = np.array(song_mean_channel)

            # Get the std of each channel
            song_std = np.std(songs[i], axis=1)
            song_std[song_std <= self.tol] = 1

            # Division by the std
            song_norm_channel = list(
                map(lambda x, y: x/y, song_mean_channel, song_std))
            song_norm_channel = np.array(song_norm_channel)

            # Save normalized spectrograms
            #print(norm_songs.shape, song_norm_channel.shape)
            norm_songs[i] = song_norm_channel
        return norm_songs


DATASET_FOLDER = ''

# Create a MelSpectral representation from the Dataset
song_rep = MelSpectrogram(DATASET_FOLDER+"Train",
                          DATASET_FOLDER+"train_clean.csv")
input_shape = (1249, 128)

songs, genres = song_rep.getdata()
songs = song_rep.normalize(songs)

print(songs.shape)
print(genres.shape)

file_pickle = 'melspectro_songs_train_new.pickle'
pickle.dump(songs, open(file_pickle, 'wb'), pickle.HIGHEST_PROTOCOL)

file_pickle = 'melspectro_genres_train_new.pickle'
pickle.dump(genres, open(file_pickle, 'wb'), pickle.HIGHEST_PROTOCOL)
