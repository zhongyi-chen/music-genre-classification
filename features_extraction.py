import librosa
import os
import numpy as np
import multiprocessing as mp
import tqdm as tq
import pandas as pd
from numpy import save
from scipy import spatial
import sys
import random 
import warnings
warnings.filterwarnings('ignore')
workingDir = os.getcwd()
print(workingDir)
os.chdir("./data")
basePath = os.getcwd()
print(basePath)

def get_mfcc(file):
  base= os.path.basename(file)
  trackId = int(os.path.splitext(base)[0])
  try:
    audio,sampleRate = librosa.load(file)
    mfcc = librosa.feature.mfcc(audio,sampleRate,n_mfcc=22)
    # convert mfcc in multiple frames into one
    mfcc = mfcc.mean(axis=1)

    obj =[trackId,mfcc]
    return obj
  except:
    return [-1,trackId]

def get_spectral_features(file):
  base= os.path.basename(file)
  trackId = int(os.path.splitext(base)[0])
  try:
    y,sr = librosa.load(file)
    
    cstft = librosa.feature.chroma_stft(y,sr);
    ccqt = librosa.feature.chroma_cqt(y,sr);
    ccens = librosa.feature.chroma_cens(y,sr);
    melsp = librosa.feature.melspectrogram(y,sr);
    mfcc = librosa.feature.mfcc(y,sr);
    rms = librosa.feature.rms(y,sr);
    spcentroid = librosa.feature.spectral_centroid(y,sr);
    spbw = librosa.feature.spectral_bandwidth(y,sr);
    spcontrast = librosa.feature.spectral_contrast(y,sr);
    spfn = librosa.feature.spectral_flatness(y);
    spro = librosa.feature.spectral_rolloff(y,sr);
    polyf = librosa.feature.poly_features(y,sr);
    tonne = librosa.feature.tonnetz(y,sr);
    zerorc = librosa.feature.zero_crossing_rate(y,sr);
    features =[cstft,ccqt,ccens,melsp,mfcc,rms,spcentroid,spbw,spcontrast,spfn,spro,polyf,tonne,zerorc]
    obj =[trackId]
    for f in features:
      obj.append(np.asarray([f.mean(axis=1),f.var(axis=1)]));
    return np.asarray(obj)
  except:
    print("Corrupted file catched track_id : ", trackId)
    return np.asarray([-1,trackId])

# This function is called for initialization
# which convert audio files to binary format that contain all information that we need
# such as mfcc
def create_mfcc(audiosPath,datasetName):
    files = librosa.util.find_files(audiosPath,ext='mp3')
    nb_files = len(files)
    p = mp.Pool(mp.cpu_count()-1)
    results=[]
    for result in tq.tqdm(p.imap(get_mfcc, files), total=nb_files):
        results.append(result)

    results=np.asarray(results)
    print("\n")
    print(results.shape)

    np.save(datasetName, np.asarray(results))


def create_features(audiosPath,datasetName):
    files = librosa.util.find_files(audiosPath,ext='mp3')
    nb_files = len(files)
    p = mp.Pool(mp.cpu_count()-1)
    results=[]
    print(datasetName)
    for result in tq.tqdm(p.imap(get_spectral_features, files), total=nb_files):
        results.append(result)

    results=np.asarray(results)
    print("\n")
    print(results.shape)

    np.save(datasetName, np.asarray(results))


trainDatasetPath = basePath+"/train/Train"
testDatasetPath = basePath+"/test/Test"
# create_features(trainDatasetPath,"train_spectral_features_dataset.npy")
create_features(testDatasetPath,"test_spectral_features_dataset.npy")
