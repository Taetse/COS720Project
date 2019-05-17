import csv
import re
import html
import urllib
import urllib.request

import cv2 as cv 
import numpy as np
import pandas as pd
# from pyagender import PyAgender
from textblob import TextBlob

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def read_from_csv(file_name: str):
    df = pd.read_csv(file_name, encoding='utf-8', sep=',', quotechar='\"', escapechar='\\', error_bad_lines=False, skipinitialspace=True)
    return df


def read_from_datbase(engine, table_name):
    a = pd.read_sql_query('select * from ' + table_name, con=engine)
    return a


def resolve_slang_and_abbreviations(df):
    df['CONTENT'] = df['CONTENT'].apply(
        lambda x: translator(x))

    print('-------Replace Abbreviations--------')
    print(df.head()['CONTENT'])


def translator(user_string):
    user_words = user_string.split(" ")

    # File path which consists of Abbreviations.
    file_name = "slang.txt"
    with open(file_name, "r") as abbreviations_csv:
        abbreviation_reader = csv.reader(abbreviations_csv, delimiter="=")
        abbreviations = {rows[0]: rows[1] for rows in abbreviation_reader}
        abbreviations_csv.close()

    for j in range(len(user_words)):
        _str = (re.sub('[^a-zA-Z0-9]+', '', user_words[j])).upper()
        if _str in abbreviations:
            user_words[j] = abbreviations[_str]

    return ' '.join(user_words)


def remove_stop_word(df):
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    
    stop = stopwords.words("english")
    df['CONTENT'] = df['CONTENT'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop]))

    print('-------Remove Stop Word--------')
    print(df.head()['CONTENT'])


def lemmatize(df):
    from nltk.stem.wordnet import WordNetLemmatizer
    lmtzr = WordNetLemmatizer()
    df['CONTENT'] = df['CONTENT'].apply(
        lambda x: ' '.join([lmtzr.lemmatize(word, 'v') for word in x.split()]))
    print('-------Lemmazation--------')
    print(df.head()['CONTENT'])


def remove_mentions(df):
    tag_pattern = re.compile(r'@[A-Za-z0-9]+')
    df['CONTENT'] = df['CONTENT'].apply(
        lambda x: tag_pattern.sub('', x))
    print('-------Tag Removal--------')
    print(df.head()['CONTENT'])


def detect_language(df):
    from langdetect import detect
    df["CONTENT_LANGUAGE"] = df['CONTENT'].apply(
        lambda x: detect(x))

    print('-------Detect Content Language--------')
    print(df.head()[['CONTENT', "CONTENT_LANGUAGE"]])


def construct_emoji_pattern():
    return re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u'\U00010000-\U0010ffff'
                      u"\u200d"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\u3030"
                      u"\ufe0f"
                      "]+", flags=re.UNICODE)


def count_emojis(df):
    emoji_pattern = construct_emoji_pattern()
    df['EMOJI_COUNT'] = df['CONTENT'].apply(
        lambda x: len(emoji_pattern.findall(x)))

    print('-------Emoji Count--------')
    print(df.head()[['CONTENT', "EMOJI_COUNT"]])


def remove_emojis(df):
    emoji_pattern = construct_emoji_pattern()
    df['CONTENT'] = df['CONTENT'].apply(
        lambda x: emoji_pattern.sub('', x))

    print('-------Remove Emojis--------')
    print(df.head()['CONTENT'])


def to_lower(df):
    df['WORD_COUNT'] = df['CONTENT'].apply(
        lambda x: len(x.split()))

    print('-------Word Count--------')
    print(df.head()[['CONTENT', "WORD_COUNT"]])

# leos shit
def extract_URLs(df):
    print("Extracting urls")
    def extract_loop(tweet):
        URLarray = []
        URL = re.search(r"(?P<url>https?://[^\s\"]+)", tweet)
        while URL != None:
            URLarray.append(URL.group("url"))
            tweet = re.sub(r"http\S+", "",tweet, 1)
            URL = re.search(r"(?P<url>https?://[^\s\"]+)", tweet)
        return URLarray
    df["URL_LIST"] = df["CONTENT"].apply(
        lambda x: extract_loop(x)
    )
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: re.sub(r"http\S+", "", x, 0)
    )
    print(df.head()[['CONTENT', "URL_LIST"]])

# not sure if needed
def escape_HTML(df):
    print("unescaping HTML chars")
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: html.unescape(x)
    )
    print(df.head()['CONTENT'])

def remove_punctuation(df):
    import string
    print("removing punctuation")
    df["CONTENT"]  = df["CONTENT"].apply(
        lambda x: "".join([char for char in x if char not in string.punctuation])
    )
    # tweet = re.sub('[0-9]+', '', tweet)
    print(df.head()['CONTENT'])

def remove_apostrophes(df):
    import apostrophes
    print("Removing apostrophes")
    df['CONTENT'] = df['CONTENT'].apply(
        lambda x: "".join([apostrophes.contractions[word]+" " if word in apostrophes.contractions else word+" " for word in x.split()])
    )
    print(df.head()['CONTENT'])

def checkSpelling(df):
    from spellchecker import SpellChecker 
    print("checking spelling")
    spell = SpellChecker(distance=2) 
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: "".join([spell.correction(word)+' ' for word in x.split()])
    )

def remove_RT(df):
    print("removing RT")
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: "".join([word+' ' if word != 'RT' else "" for word in x.split()])
    )

def get_sentiment(df):
    df['SENTIMENT'] = df['CONTENT'].apply(
        lambda x: TextBlob(x).sentiment.polarity)

    print('-------Sentiment Analysis--------')
    print(df.head()[['CONTENT', "SENTIMENT"]])


# def url_to_image(url):
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv.imdecode(image, cv.IMREAD_COLOR)

#     return image


# def detect_face(url):
#     try:
#         image = url_to_image(url)
#     except urllib.error.HTTPError:
#         return False

#     grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     faceCascade = cv.CascadeClassifier(r'data-cleaning\classifier.xml')
#     faces = faceCascade.detectMultiScale(grayscale_image)
#     if len(faces) > 0:
#         return True
#     else:
#         return False


# def facial_recognition(df):
#     df['PFP_CONTAIN_FACE'] = df['PROFILE_IMAGE'].apply(
#         lambda x: detect_face(x))

#     print('-------Face Recognition--------')
#     print(df.head()[['CONTENT', "PFP_CONTAIN_FACE"]])


# def get_estimate_age(url):
#     agender = PyAgender()
#     image = url_to_image(url)
#     faces = agender.detect_genders_ages(image)
#     return round(faces[0]['age'])


# def estimate_age(df):
#     df['ESTIMATE_AGE'] = df.loc[df['PFP_CONTAIN_FACE'] == 'True']['PROFILE_IMAGE'].apply(
#         lambda x: get_estimate_age(x))

#     print('-------Estimate Age--------')
#     print(df.head()[['CONTENT', "ESTIMATE_AGE"]])


def main():
    df = read_from_csv(r"D:\UP\COS\720\repo\shortened-data.csv")
    
    print("--- Print the Head of the data ---")
    print(df.head()["CONTENT"])

    # detect_language(df)
    remove_RT(df)
    escape_HTML(df) # not sure if needed
    remove_mentions(df)
    count_emojis(df)
    remove_emojis(df)
    extract_URLs(df)
    remove_apostrophes(df)
    remove_punctuation(df)
    resolve_slang_and_abbreviations(df)
    checkSpelling(df) # expensive task
    remove_stop_word(df)
    lemmatize(df)
    to_lower(df)
    get_sentiment(df)
    # facial_recognition(df)
    # estimate_age(df)


if __name__ == '__main__':
    main()
