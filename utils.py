import numpy as np
import pandas as pd
import random
import tensorflow.keras
num_classes = 8
def create_submission(ids,labels,corrupted_ids,name):
    dataset = np.stack([ids,labels],axis=1)
    filename = name+".csv"
    for c in corrupted_ids:
        new = np.array([[c,random.randint(1,8)]])
        dataset=np.append(dataset,new,axis=0)
    dataset = np.int_(sorted(dataset,key=lambda x: x[0]))
    results = pd.DataFrame(data = {'track_id': dataset[:,0], 'genre_id': dataset[:,1]}, columns = ['track_id', 'genre_id'])
    results.to_csv(filename, index=False)
#     print(filename + "created")
    
    
def vgg_train_dataset():
    _songs = np.load("./data/vgg/trainvgg_dataset.npy")
    X =  _songs.reshape(_songs.shape[0],_songs.shape[1],_songs.shape[2],1)
    
    _ids = np.load("./data/vgg/trainvgg_ids.npy")
    traingenre = pd.read_csv(filepath_or_buffer="./data/train_clean.csv", sep=",")
    
    track_ids = pd.DataFrame(data = {'track_id':_ids}, columns = ['track_id'])
    train_dataset = pd.merge(track_ids, traingenre, on='track_id')
    
    Y = train_dataset['genre_id'].values
    Y = tensorflow.keras.utils.to_categorical(Y-1, num_classes)
    
    return X,Y

def vgg_train_dataset_ps():
    _songs = np.load("./data/vgg/trainvgg_ps_dataset.npy")
    X =  _songs.reshape(_songs.shape[0],_songs.shape[1],_songs.shape[2],1)
    
    _unique_ids = np.load("./data/vgg/trainvgg_ps_ids.npy")
    _ids=[]
    for i in _unique_ids:
        for k in range(3):
            _ids.append(i)
    _ids = np.asarray(_ids)
    traingenre = pd.read_csv(filepath_or_buffer="./data/train_clean.csv", sep=",")
    print(_songs.shape, _ids.shape)
    track_ids = pd.DataFrame(data = {'track_id':_ids}, columns = ['track_id'])
    train_dataset = pd.merge(track_ids, traingenre, on='track_id')
    
    Y = train_dataset['genre_id'].values
    Y = tensorflow.keras.utils.to_categorical(Y-1, num_classes)
    
    return X,Y

def vgg_test_dataset():
    _songs = np.load("./data/vgg/testvgg_dataset.npy")
    X =  _songs.reshape(_songs.shape[0],_songs.shape[1],_songs.shape[2],1)
    track_ids = np.load("./data/vgg/testvgg_ids.npy")
    corrupted = np.load("./data/vgg/testvgg_corrupted.npy")
    return X, track_ids, corrupted

def vgg_test_dataset_ps():
    _songs = np.load("./data/vgg/testvgg_ps_dataset.npy")
    X =  _songs.reshape(_songs.shape[0],_songs.shape[1],_songs.shape[2],1)
    _unique_track_ids = np.load("./data/vgg/testvgg_ps_ids.npy")
    _ids=[]
    for i in _unique_track_ids:
        for k in range(3):
            _ids.append(i)
    _ids = np.asarray(_ids)
    corrupted = np.load("./data/vgg/testvgg_ps_corrupted.npy")
    return X, _ids, corrupted