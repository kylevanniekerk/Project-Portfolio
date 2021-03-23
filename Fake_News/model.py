# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 06:06:00 2021

@author: Admin
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import re
import matplotlib.pyplot as plt
import itertools
import pickle

train = pd.read_csv('/home/kyle/Projects/Fake_News/train.csv', index_col = 'id')
null = train.isnull().sum()
train[train['text'].isna()].label
train[train['author'].isna() & train['label'] == 1].label.sum()
train = train.fillna(0)

df = train.copy()
df = df.drop('title', axis = 1)

def tfidfvect(df):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, len(df)):
        text = re.sub('[^a-zA-Z]', ' ', str(df['text'][i])).lower().split()
        text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
        corpus.append(text)      
    return corpus

def labelencode(col):
    name = []
    le = LabelEncoder()
    for i in range(0, len(col)):
        auth = re.sub('[^a-zA-Z]', ' ', str(col[i])).lower().split()
        auth = ' '.join(auth)
        name.append(auth)
    le.fit(name)
    X = le.transform(name)
    return X

authors = df['author']       
encode_auth = labelencode(authors)
tfidfv = TfidfVectorizer(max_features = 10000, ngram_range=(1, 3))
vect = tfidfv.fit_transform(tfidfvect(df)).toarray()
X = pd.DataFrame(vect)
X['author'] = encode_auth
y = df['label']

pickle.dump(tfidfv, open('Vectorize.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)
model = XGBClassifier(n_estimators = 1000, learning_rate = 0.05, n_jobs = 4)
model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(X_test, y_test)], verbose = False)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

pred = model.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print('Accuracy: %0.3f' % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

pickle.dump(model, open('model.pkl', 'wb'))










            


        