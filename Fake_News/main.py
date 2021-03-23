#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:38:13 2021

@author: kyle
"""
import uvicorn 
from fastapi import FastAPI
from model import tfidfvect
import pickle 
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#import numpy as np 
from pydantic import BaseModel

# Create app object
app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))
tfidfv = pickle.load(open('Vectorize.pkl', 'rb'))

# Class that describes the user input values
class FakeNews(BaseModel):
    text_body: str

@app.get("/")
def root():
    return {'This a Fake News Classifier done by Kyle van Niekerk'}

# Expose the prediction functionality, make a prediction from the passed JSON data and return label with confidence. 
@app.post("/predict")
def predict_fake_news(text: FakeNews):
    corpus = []
    stemmer = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
    vect = tfidfvect(text)
    X = tfidfv.transform(vect).toarray()
    X = pd.DataFrame(X)
    prediction = model.predict(X)
    
    if prediction > 0.5:
        prediction = 'This is an unreliable article'
    else:
        prediction = 'This article is reliable'
    return prediction
    

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)
# uvicorn main:app --reload