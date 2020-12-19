import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm as tq
import os
import warnings
warnings.filterwarnings('ignore')
workingDir = os.getcwd()
print(workingDir)
os.chdir("../data")
basePath = os.getcwd()
trainDatasetPath = basePath+"/train/Train"
testDatasetPath = basePath+"/test/Test"
# Load the model.
vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

def embedding_from_fn(fn):
    base= os.path.basename(fn)
    trackId = int(os.path.splitext(base)[0])
    try:
        x, sr = librosa.load(fn) #,sr=None
        x_16k = librosa.resample(x,sr,16000) #resample to 16KHz
        x_ps4 = librosa.effects.pitch_shift(x_16k, sr, n_steps=4)
        x_ps6 = librosa.effects.pitch_shift(x_16k, sr, n_steps=6)
        embedding = np.array(vggmodel(x_16k)) 
        embedding_ps4 = np.array(vggmodel(x_ps4)) 
        embedding_ps6 = np.array(vggmodel(x_ps6)) 
        return trackId, embedding, embedding_ps4, embedding_ps6
    except:
        print("Corrupted file catched track_id : ", trackId)
        return -1,trackId,-1,-1
    
def extract(trainDatasetPath,name):
    files = librosa.util.find_files(trainDatasetPath,ext='mp3')
    dataset=[]
    ids = []
    corrupted =[]
    for f in tq.tqdm(files):
        track_id,embedding,embedding_ps4,embedding_ps6 = embedding_from_fn(f)
        if(track_id==-1):
            corrupted.append(embedding)
        else:
            if(embedding.shape[0]==31 and embedding.shape[1]==128):
                ids.append(track_id)
                dataset.append(embedding)
                dataset.append(embedding_ps4)
                dataset.append(embedding_ps6)
            else:
                print("Shape error : " , embedding.shape)
                corrupted.append(track_id)

    dataset=np.asarray(dataset)
    ids=np.asarray(ids)
    corrupted=np.asarray(corrupted)
    print("\n")
    print(dataset.shape, ids.shape, corrupted.shape)
    np.save("vgg/"+name+"vgg_ps_dataset.npy", dataset)
    np.save("vgg/"+name+"vgg_ps_ids.npy", ids)
    np.save("vgg/"+name+"vgg_ps_corrupted.npy", corrupted)
    
extract(trainDatasetPath,"train")
# extract(trainDatasetPath,"test")