import numpy as np
import pandas as pd


## corrupted files 098559  098571  Test
## corrupted files 105247  126981  133297  Train

# https://arxiv.org/pdf/1612.01840.pdf
partiel_feature_switcher={
  # 'chroma_stft' :1,
  # 'chroma_cqt' :2,
  # 'chroma_cens' :3,
  # 'mel' :4,
  'mfcc' :5,
  'rms' :6,
  'spectral_centroid' :7,
  'spectral_bandwidth' :8,
  'spectral_contrast' :9,
  'spectral_flatness' :10,
  'spectral_rolloff' :11,
  # 'poly_features' :12,
  'tonnetz' :13,
  'zero_crossing_rate' :14,

}
## features_file : binary npy format
## feature_label_file : csv format or None
def process_dataset(features_file,feature_label_file=None):
  features = np.load(features_file,allow_pickle=True)
  feature_label_arr =  pd.read_csv(feature_label_file).values if feature_label_file else []

  dataset =[]
  corrupted_id_arr=[]
  for t in features:
    id = t[0]
    if(id!=-1):
      ## concatenate features
      features=[]
      if(feature_label_file):
        label = feature_label_arr[feature_label_arr[:,0]==id][0][1]
        features.append(label)
      for fs in partiel_feature_switcher:
        index  = partiel_feature_switcher.get(fs)
        features = np.concatenate((features,t[index][0,:],t[index][1,:]), axis=None)
      dataset.append(features)
    else:
      corrupted_id_arr.append(t[1])
  return np.asarray(dataset),np.asarray(corrupted_id_arr)


def partial_dataset():
  train_features_file = "./data/train_spectral_features_dataset.npy"
  train_feature_label_file = "./data/train.csv"
  train_dataset,train_corrupted_id = process_dataset(train_features_file,train_feature_label_file)

  test_features_file = "./data/test_spectral_features_dataset.npy"
  test_dataset,test_corrupted_id = process_dataset(test_features_file)
  # fisrt column corresponds to genre then the all rest is features according to partiel_feature_switcher
  return train_dataset,test_dataset,test_corrupted_id

# The following code and function based on features_adapte dataset
# In total of 106574 musics

# Each descriptor is present with 
# key = name 
# value = (index, size of features)
full_feature_switcher={
  'chroma_cens' :(1,12),
  'chroma_cqt' :(2,12),
  'chroma_stft' :(3,12),
  'mfcc' :(4,20),
  'rms' :(5,1),
  'spectral_bandwidth' :(6,1),
  'spectral_centroid' :(7,1),
  'spectral_contrast' :(8,7),
  'spectral_rolloff' :(9,1),
  'tonnetz' :(10,6),
  'zero_crossing_rate' :(11,1),
}

statistic_switcher={
  'kurtosis' :1,
  'max':2,
  'mean':3,
  'median':4,
  'min':5,
  'skew':6,
  'std':7
}

statistic_switcher_to_drop={
  # 'kurtosis' :1,
  'max':2,
  # 'mean':3,
  # 'median':4,
  'min':5,
  # 'skew':6,
  # 'std':7
}

def new_column_names():
  names=['track_id']
  for f in full_feature_switcher:
    size = full_feature_switcher.get(f)[1]
    for s in statistic_switcher:
      for i in range(size):
        names.append(f+'_'+s+'_'+str(i))
  return names

def get_to_drop_list():
  to_drop=[]
  for f in full_feature_switcher:
    size = full_feature_switcher.get(f)[1]
    for s in statistic_switcher_to_drop:
      for i in range(size):
        to_drop.append(f+'_'+s+'_'+str(i))
  return to_drop

def filter_full_dataset():
  features = pd.read_csv(filepath_or_buffer="./data/features_head.csv", sep=",")
  traingenre = pd.read_csv(filepath_or_buffer="./data/train_clean.csv", sep=",")
  iter_csv = pd.read_csv(filepath_or_buffer="./data/features_adapte.csv", sep=",", iterator=True, chunksize=10000)

  for chunk in iter_csv:
    # selected by column index
    print(chunk.iloc[:,1])

def full_dataset():
  features = pd.read_csv(filepath_or_buffer="./data/features_head.csv", sep=",")
  traingenre = pd.read_csv(filepath_or_buffer="./data/train_clean.csv", sep=",")
  test = pd.read_csv(filepath_or_buffer="./data/test.csv", sep=",")
  iter_csv = pd.read_csv(filepath_or_buffer="./data/features_adapte.csv", sep=",", iterator=True, chunksize=10000)
  datatrain = pd.concat([chunk for chunk in iter_csv])
  # assign new column titles 
  new_col_names = new_column_names()
  datatrain = datatrain.set_axis(new_col_names,axis=1)
  # remove unwanted features
  to_drop_list = get_to_drop_list()
  datatrain =datatrain.drop(to_drop_list,axis=1)

  train_dataset = pd.merge(traingenre, datatrain, on='track_id')
  test_dataset = pd.merge(test, datatrain, on='track_id')
  print(train_dataset.shape,test_dataset.shape, traingenre.shape, datatrain.shape)
  X_train = train_dataset.drop(['track_id','genre_id'],axis=1).values
  y_train = train_dataset['genre_id'].values
  # remove corrupted file
  print(test_dataset.shape, "test shape")
  arr =[098559.0,098571.0]
  test_dataset = test_dataset.drop(test_dataset[test_dataset.track_id == 098559.0].index)
  test_dataset = test_dataset.drop(test_dataset[test_dataset.track_id == 098571.0].index)
  print(test_dataset.shape, "test shape")

  X_test = test_dataset.drop('track_id',axis=1).values
  print(X_train.shape, y_train.shape,X_test.shape)
  return X_train, X_test, y_train

# full_dataset()