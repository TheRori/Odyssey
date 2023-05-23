import logging
import pandas as pd
from pathlib import Path
import json
import re
import sys
import senticnet
import tagtog2df.tagtog2df as tagtog2df
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from pyFeel import Feel

# Load data
path = Path('json/members')
df = tagtog2df.allfiles_onedataframe(path)

# Remove columns with e_14, e_25, r_26
df = df[df['Class ID'] != 'e_14'] # speaker
df = df[df['Class ID'] != 'e_25'] # speech
df = df[df['Class ID'] != 'r_26'] # vb2yp6dteni(e_14|e_25)

# Keep only text
df = df[['Text', 'Class ID']]
df = df.rename(columns={'Text': 'text', 'Class ID': 'human'})

# Insert columns
df.insert(0, 'id', [i for i in range(len(df))] )
df.insert(2, 'text_tokenized', None)
df.insert(3, 'senticnet', None)
df.insert(4, 'pyfeel', None)

# Load French stopwords
stopwords = list(set(stopwords.words('french')))
punctuation = [',', '.', ':', ';', '(', ')', '[', ']', '!', '?', '+', '-', '*', "'", '"', '/']

# Load SenticNet dictionary
stcnet = senticnet.Senticnet()

def senticnetEmotion(stcnet, tokens):
    """
        Average emotion of a list of tokens
    """
    
    emotions = []
    emotions_dict = {}
    
    # Retrieve emotions data for all the tokens of the sentence
    for token in tokens:
        primary_secondary_emotions = stcnet.emotionsOf(token)
        
        # Appends to emotions
        for e in primary_secondary_emotions:
            if primary_secondary_emotions[e]:
                emotions.append(primary_secondary_emotions[e])

    # Counts emotions
    for emotion in set(emotions):
        emotions_dict[emotion] = emotions.count(emotion)

    # Finds emotion of speech (max)
    emotion_of_speech = None
    if emotions_dict:
        emotion_of_speech = max(emotions_dict, key=emotions_dict.get)

    return emotion_of_speech

def pyfeelEmotion(text):
    """
        Retrieve most likely emotion based on pyfeel algorithm
    """
    
    # Run algorithm    
    pfe = Feel(text)
    pfe = pfe.emotions()

    # Find max emotion
    if pfe:
        emotion_of_speech = max(pfe, key=pfe.get)
        
    # TO DO !!
    
    return emotion_of_speech

for rid, row in df.iterrows():
    row_id = row['id']
    text = row['text']
    tokens = word_tokenize(text)
    tokens_list = []
    
    # Processing for SenticNet
    for token in tokens:
        token = token.lower().strip()
        # Replace ' by nothing to match SenticNet
        # and - by _
        token = token.replace("'", '')
        token = token.replace('-', '_')
        
        # Replace spaces and - by _
        token = re.sub(r'\s+', '_', token)
        
        # Replace accents and normalize text
        token = unidecode(token)
    
        # Avoid stopwords and punctuation marks
        if token not in stopwords and token not in punctuation and token:
            # Append to list
            tokens_list.append(token)

    # Find SenticNet Emotion and updates dataframe
    senticnet_emotion = senticnetEmotion(stcnet, tokens_list)
    
    # Update df
    df.loc[df['id']== row_id, 'senticnet'] = senticnet_emotion
    df.loc[df['id']== row_id, 'text_tokenized'] = json.dumps(tokens_list)

    # Find PyFeel emotions
    #pyfeel_emotion = pyfeelEmotion(text)

# Replace emotions' code by their human readable labels
emotions_tag = {
  'e_8'  : 'fear',
  'e_3'  : 'joy',
  'e_27' : 'calmness',
  'e_7'  : 'angry',
  'e_5'  : 'disgust',
  'e_6'  : 'sadness',
  'e_4'  : 'surprise'
}

for emotion in emotions_tag:
    df['human'] = df['human'].replace(to_replace=emotion, value=emotions_tag[emotion])

# Exports data
df.to_csv('emotions.csv')
