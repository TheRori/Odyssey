# This is a sample Python script.
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


import tagtog2df
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tagtog2df.tagtog2df import allfiles_onedataframe
import senticnet
from pyfeel.pyFeel import Feel

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
    # Use a breakpoint in the code line below to debug your script.
    path = Path('jsons/members')
    df_row = allfiles_onedataframe(path)
    df_clean = df_row[['Class ID', 'Text']]
    df_clean = df_clean.rename(columns={"Class ID": "TagTog"})
    df_clean = df_clean.loc[(df_clean["TagTog"] != 'e_14') & (df_clean["TagTog"] != 'e_25') &
                            (df_clean["TagTog"] != 'e_1') & (df_clean["TagTog"] != 'e_2') &
                            (df_clean["TagTog"] != 'e_9')]
    df1 = df_clean.reset_index(drop=True)
    df1 = df1.replace('e_8', annotation['e_8'])
    df1 = df1.replace('e_3', annotation['e_3'])
    df1 = df1.replace('e_27', annotation['e_27'])
    df1 = df1.replace('e_7', annotation['e_7'])
    df1 = df1.replace('e_5', annotation['e_5'])
    df1 = df1.replace('e_6', annotation['e_6'])
    df1 = df1.replace('e_4', annotation['e_4'])

    return df1


def add_emotions_senticnet(df):
    emotions = []
    for l in df['Text']:
        emotions.append(s.averageEmotionsOf(l.split())['primary_emotion'])
    df['Senticnet'] = emotions


def add_emotions_pyfeel(df):
    emotions = []
    for l in df['Text']:
        em = Feel(l)
        emotions.append(max(em.emotions(), key=em.emotions().get))
    df['PyFeel'] = emotions


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # df = make_df()
    # add_emotions_senticnet(df)
    # add_emotions_pyfeel(df)
    # df.to_csv('csv/emotions_df.csv')
    df = pd.read_csv('csv/emotions_df.csv')
    df['TagTog'].value_counts().plot(kind='bar')
    plt.show()
    df['Senticnet'].value_counts().plot(kind='bar')
    plt.show()
    df['PyFeel'].value_counts().plot(kind='bar')
    plt.show()
    print(df.TagTog.unique())
    print(df.Senticnet.unique())
    print(df.PyFeel.unique())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
