# Music genre classification

We try to classify music genre by machine learning approach. In this projet, 4000 musics are used training and then we have to classify other 4000 test musics.

Accuracy with half test data on Kaggle:

| Method                            | Accuracy  |
| :-------------------------------- | :-------: |
| MFCC similarity                   |    33%    |
| KNN 11 Features                   |    48%    |
| LR 11 Features                    |    55%    |
| MLP 11 Features                   |    56%    |
| SVM 11 Features                   |    58%    |
| CatBoost 11 Features              |    60%    |
| CatBoost 11 Features Grid Seach   |    64%    |
| CNN Mel-spectrogram               |    39%    |
| VGGish + CNN                      |   64.8%   |
| VGGish + CRNN                     | **67.9%** |
| VGGish + CNN with pitch shifting  |   64.3%   |
| VGGish + CRNN with pitch shifting |   67.8%   |

We obtient a precision of **68.7%** on Kaggle with `VGGish + CRNN` method by adjusting the train and validation ratio to 0.9 and some tuning on hyperparameter such as MaxPooling size. Finally, the precision drop to **66.2%** with another half test data on Kaggle.

[Link](https://www.kaggle.com/c/tsma-202021-music-genre-classification/overview) to Kaggle competition and training data.
