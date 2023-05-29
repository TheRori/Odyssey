# This is a sample Python script.
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from unidecode import unidecode
from matplotlib_venn import venn2, venn3
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

def textsPerEmotions(m, em):
    return len(df[df[m] == em])

def make_diagram(df,em):
    inter1 = df[(df['TagTog'] == em) & (df['PyFeel'] == em)]
    inter2 = df[(df['Senticnet'] == em) & (df['PyFeel'] == em)]
    inter3 = df[(df['Senticnet'] == em) & (df['TagTog'] == em)]
    inter4 = df[(df['Senticnet'] == em) & (df['TagTog'] == em) & (df['PyFeel'] == em)]

    n1 = textsPerEmotions('TagTog',em)
    n2 = textsPerEmotions('PyFeel',em)
    n3 = textsPerEmotions('Senticnet',em)

    df_similarity = pd.DataFrame(np.array([[em,n1,n2,n3,(len(inter1)),(len(inter3)),(len(inter2)),(len(inter4))]]),
                 columns=['Emotion','TagTog', 'PyFeel', 'Senticnet','T-P','T-S','P-S','T-P-S'])

    venn3(subsets=(n1, n2, n3, len(inter1), len(inter2), len(inter3), len(inter4)),
          set_labels=('TagTog', 'PyFeel', 'Senticnet'))
    plt.title(em)
    plt.show()
    return df_similarity

def normalizeEmSenticnet(df):
    df = df.apply(lambda x: x.astype(str).str.lower())
    df = df.replace('ecstasy','joy')
    df = df.replace('contentment','joy')
    df = df.replace('anxiety','fear')
    df = df.replace('terror','fear')
    df = df.replace('dislike','disgust')
    df = df.replace('loathing','disgust')
    df = df.replace('bliss','calmness')
    df = df.replace('serenity','calmness')
    df = df.replace('grief','sadness')
    df = df.replace('melancholy','sadness')
    df = df.replace('anger','angry')
    df = df.replace('annoyance','angry')
    df = df.replace('rage','angry')
    df = df.replace('acceptance','pleasantness')
    df = df.replace('delight','pleasantness')
    df = df.replace('enthusiasm','eagerness')
    df = df.replace('responsiveness','eagerness')
    return df


if __name__ == '__main__':

    #df = pd.read_csv('csv/all_df_clean.csv')

    #df = make_df()
    #add_emotions_senticnet(df)

    #add_emotions_pyfeel(df)
    #df.to_csv('csv/emotions_df.csv')
    df = pd.read_csv('csv/emotions_df.csv')
    df = normalizeEmSenticnet(df)
    emotions = (list(df.Senticnet.unique()))
    emotions.extend(list(df.TagTog.unique()))
    emotions.extend(list(df.PyFeel.unique()))
    emotions = list(dict.fromkeys(emotions))
    df2 = pd.DataFrame(
                 columns=['Emotion','TagTog', 'PyFeel', 'Senticnet','T-P','T-S','P-S','T-P-S'])
    for e in emotions:
        df1 = make_diagram(df,e)
        df2 = pd.concat([df2, df1], axis=0)
    df2 = df2.reset_index(drop=True)
    df2.to_csv('csv/stats_emotions.csv')
    # df3 = make_diagram(df,'Calmness','calmness','grief')
    # df4 = make_diagram(df,'Joy','joy','serenity')
    # dfc= pd.concat([df2, df3], axis=0)
    # dfc= pd.concat([dfc, df4], axis=0)
    #
    # print(dfc.head())
    #df['TagTog'].value_counts().plot(kind='bar')
    #plt.show()
    #df['Senticnet'].value_counts().plot(kind='bar')
    #plt.show()
    #df['PyFeel'].value_counts().plot(kind='bar')
    #plt.show()
    # print(df.TagTog.unique())
    # print(df.Senticnet.unique())
    # print(df.PyFeel.unique())
