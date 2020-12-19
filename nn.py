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
workingDir = os.getcwd()
print(workingDir)
os.chdir("./data")
workingDir = os.getcwd()
print(workingDir)

# load bianry train and test dataset
trainDataSet = np.load("train_mfcc_dataset.npy",allow_pickle=True)
trainDataSet = trainDataSet[trainDataSet[:,0]!=-1]
traintLabels = pd.read_csv('train.csv',converters={'track_id': lambda x: str(x)}).values
# print(trainDataSet[0])
# print(traintLabels[0])
testDataSet = np.load("test_mfcc_dataset.npy",allow_pickle=True)
test_corrupted = testDataSet[testDataSet[:,0]==-1]
testDataSet = testDataSet[testDataSet[:,0]!=-1]
testtLabels = pd.read_csv('test.csv',converters={'track_id': lambda x: str(x)}).values
print(testtLabels.shape)
test_csv= np.c_[testtLabels,np.zeros(testtLabels.shape[0])]

for corrupted in test_corrupted:
   index = np.where(test_csv[:,0]==corrupted[1])
   test_csv[index,1]= random.randint(1,8)



for test in tq.tqdm(testDataSet):
  distance = 9999999
  genreBestMatch=-1
  for train in trainDataSet:
    d = spatial.distance.cosine(test[1],train[1])
    l,genre = traintLabels[traintLabels[:,0]==train[0]][0]
    if d<distance:
      distance=d
      genreBestMatch = genre
  index = np.where(test_csv[:,0]==test[0])
  test_csv[index,1]= genreBestMatch

results = pd.DataFrame(data = {'track_id': test_csv[:,0], 'genre_id': test_csv[:,1]}, columns = ['track_id', 'genre_id'])
  
results.to_csv("results_mfcc_nn.csv", index=False)
print("done")