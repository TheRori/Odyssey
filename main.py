# This is a sample Python script.
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tagtog2df.tagtog2df import allfiles_onedataframe
import senticnet
from pyfeel.pyFeel import Feel


stopwords_lst = list(set(stopwords.words('french')))
stopwords_lst.extend(('l','le','la','d','m'))

punctuation = [',', '.', ':', ';', '(', ')', '[', ']', '!', '?', '+', '-', '*', "'", '"', '/']

annotation = {
    "e_8": "Fear",
    "e_3": "Joy",
    "e_27": "Calmness",
    "e_7": "Angry",
    "e_14": "Speaker",
    "e_5": "Disgust",
    "e_6": "Sadness",
    "e_4": "Surprise",
    "e_25": "Speech"
}

s = senticnet.Senticnet(path='senticnet/senticnet.py')


def make_df():
    path = Path('jsons/members')
    df_row = allfiles_onedataframe(path)
    df_clean = df_row[['Class ID', 'Text']]
    df_clean = df_clean.rename(columns={"Class ID": "TagTog"})
    df_clean = df_clean.loc[(df_clean["TagTog"] != 'e_14') & (df_clean["TagTog"] != 'e_25') &
                            (df_clean["TagTog"] != 'e_1') & (df_clean["TagTog"] != 'e_2') &
                            (df_clean["TagTog"] != 'e_9')]

    df1 = df_clean.reset_index(drop=True)

    mask = df1['Text'].str.strip().str.split(' ').str.len().eq(1)
    df1 = df1[~mask]
    df1 = df1.reset_index(drop=True)
    df1 = df1.replace('e_8', annotation['e_8'])
    df1 = df1.replace('e_3', annotation['e_3'])
    df1 = df1.replace('e_27', annotation['e_27'])
    df1 = df1.replace('e_7', annotation['e_7'])
    df1 = df1.replace('e_5', annotation['e_5'])
    df1 = df1.replace('e_6', annotation['e_6'])
    df1 = df1.replace('e_4', annotation['e_4'])

    df1.to_csv('csv/all_df_clean.csv')

    return df1


def add_emotions_senticnet(df):
    emotions = []
    for t in df['Text']:
        token = []
        t = t.lower().strip()
        t = t.replace("'", ' ')
        t = t.replace('-', '_')
        t = word_tokenize(t)
        for w in t:
            if w not in stopwords_lst and w not in punctuation:
                token.append(unidecode(w))
        emotions.append(s.averageEmotionsOf(token)['primary_emotion'])
    df['Senticnet'] = emotions


def add_emotions_pyfeel(df):
    emotions = []
    for l in df['Text']:
        e = Feel(l)
        em = e.emotions()
        if em.get('sadness') == 0 and em.get('disgust') == 0 and em.get('fear') == 0 and em.get('surprise') == 0 and em.get('angry') == 0 and em.get('joy') == 0 :
            emotions.append('calmness')
            continue
        max_em = max(em, key=em.get)
        if max_em == 'positivity':
            del em['positivity']
            max_em = max(em, key=em.get)
        emotions.append(max_em)
    df['PyFeel'] = emotions


if __name__ == '__main__':

    #df = pd.read_csv('csv/all_df_clean.csv')

    df = make_df()
    add_emotions_senticnet(df)

    add_emotions_pyfeel(df)
    df.to_csv('csv/emotions_df.csv')
    df = pd.read_csv('csv/emotions_df.csv')
    df['TagTog'].value_counts().plot(kind='bar')
    plt.show()
    df['Senticnet'].value_counts().plot(kind='bar')
    plt.show()
    df['PyFeel'].value_counts().plot(kind='bar')
    plt.show()
    # print(df.TagTog.unique())
    # print(df.Senticnet.unique())
    # print(df.PyFeel.unique())
