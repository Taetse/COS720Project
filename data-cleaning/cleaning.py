import csv
import re
import html
import urllib
import urllib.request

import cv2 as cv
import numpy as np
import pandas as pd
from pyagender import PyAgender
from textblob import TextBlob
from datetime import datetime
import requests
import safebrowsing

from sklearn import datasets
from sklearn.cluster import KMeans

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
    file_name = "data-cleaning\slang.txt"
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


def word_count(df):
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
            tweet = re.sub(r"http\S+", "", tweet, 1)
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
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: html.unescape(x)
    )

    print('-------Escape HTML Chars--------')
    print(df.head()['CONTENT'])


def to_lower(df):
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: x.lower()
    )
    print('-------To Lower--------')
    print(df.head()[['CONTENT']])


def remove_punctuation(df):
    import string
    print("removing punctuation")
    df["CONTENT"] = df["CONTENT"].apply(
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
    spell = SpellChecker(distance=10)
    df["CONTENT"] = df["CONTENT"].apply(
        lambda x: "".join([spell.correction(word)+' ' for word in x.split()])
    )


def get_sentiment(df):
    df['SENTIMENT'] = df['CONTENT'].apply(
        lambda x: TextBlob(x).sentiment.polarity)

    print('-------Sentiment Analysis--------')
    print(df.head()[['CONTENT', "SENTIMENT"]])


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image


def detect_face(url):
    try:
        image = url_to_image(url)
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier(r'data-cleaning\classifier.xml')
        faces = face_cascade.detectMultiScale(grayscale_image)
        if len(faces) > 0:
            return True
        else:
            return False
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False
    except:
        return False


def facial_recognition(df):
    df['PFP_CONTAIN_FACE'] = df.apply(
        lambda x: False if x["IS_DEFAULT_PROFILE"] else detect_face(x["PROFILE_IMAGE"]), "columns"
    )

    print('-------Face Recognition--------')
    print(df.head()[['CONTENT', "PFP_CONTAIN_FACE"]])


def get_estimate_age(url):
    try:
        image = url_to_image(url)
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError:
        return False
    except:
        return False

    agender = PyAgender()
    faces = agender.detect_genders_ages(image)
    if len(faces) > 0:
        return round(faces[0]['age'])
    else:
        return 0


def estimate_age(df):
    df['ESTIMATE_AGE'] = df.loc[df['PFP_CONTAIN_FACE']]['PROFILE_IMAGE'].apply(
        lambda x: get_estimate_age(x))

    print('-------Estimate Age--------')
    print(df.head()[['CONTENT', "ESTIMATE_AGE"]])


def k_means_prediction(df):
    # Declaring Model
    model = KMeans(n_clusters=2)
    # Fitting Model
    # model.fit(df[["SENTIMENT", "PFP_CONTAIN_FACE", "ESTIMATE_AGE", "WORD_COUNT", "EMOJI_COUNT", "CONTENT_LANGUAGE",
    # "TIME_AFTER_PFP_CREATION", "TWEET_LANG_SAME_PROFILE_LANG", "SOURCE", "IS_DEFAULT_PROFILE", "STATUS_COUNT", "TRANSLATOR",
    # "RTFOLLOWERS", "FRIENDS", "FOLLOWERS", "LANGUAGE", "GEO_ENABLED", ""]])
    model.fit(df[["SENTIMENT", "WORD_COUNT", "EMOJI_COUNT"]])

    df['CLUSTER'] = df.apply(
        lambda x: model.predict([[x["SENTIMENT"], x["WORD_COUNT"], x["EMOJI_COUNT"]]]), axis=1)

    print('-------K Means Clusters--------')
    print(df.head()[['CONTENT', "CLUSTER"]])


def tweetlang_in_lang(tweet_lang, lang):
    # print(str(tweet_lang) + " in " + str(lang) + " " + str(tweet_lang in lang))
    try:
        return tweet_lang in lang
    except TypeError:
        return False


def is_tweet_language_profile_language(df):
    df['TWEET_LANG_SAME_PROFILE_LANG'] = df.apply(
        lambda x: tweetlang_in_lang(x["CONTENT_LANGUAGE"], x["LANGUAGE"]), axis=1)

    print('-------Tweet language same as Profile Language--------')
    print(df.head()[['CONTENT', "TWEET_LANG_SAME_PROFILE_LANG"]])


def time_after_profile_creation(df):
    FMT = "%Y-%m-%d %H:%M:%S.%f0000"

    df['TIME_AFTER_PFP_CREATION'] = df.apply(
        lambda x: datetime.strptime(x["CREATEDAT"], FMT) - datetime.strptime(x["OPEN_DATE"], FMT), axis=1)

    print('-------Time after Profile Creation--------')
    print(df.head()[['CONTENT', "TIME_AFTER_PFP_CREATION"]])


key = 'AIzaSyAYeCUJwGYBKRdvifnR3ggtuR12t0xe3vA'
URL = "https://sb-ssl.google.com/safebrowsing/api/lookup?client=api&apikey={key}&appver=1.0&pver=3.0&url={url}"


def is_safe(urls):
    for url in urls:
        response = requests.get(URL.format(key=key, url=url))
        return response.text != 'malware'


apikey = 'AIzaSyAYeCUJwGYBKRdvifnR3ggtuR12t0xe3vA'
sb = safebrowsing.LookupAPI(apikey)


def is_phising_links(links):
    for link in links:
        resp = sb.threat_matches_find(link)
        if len(resp["matches"]) > 0:
            return True
    return False


def is_phising_site(df):
    df['CONTAINS_PHISING'] = df["URL_LIST"].apply(
        lambda x: is_phising_links(x))

    print('-------Tweet language same as Profile Language--------')
    print(df.head()[['CONTENT', "TWEET_LANG_SAME_PROFILE_LANG"]])


def main():
    # df = read_from_csv(r"C:\Users\myron\Downloads\test-data.csv")
    df = read_from_csv(r"C:\Users\myron\Downloads\Book1.csv")

    print("--- Print the Head of the data ---")
    print(df.head()["CONTENT"])

    detect_language(df)  # expensive task
    escape_HTML(df)  # not sure if needed
    remove_mentions(df)
    count_emojis(df)
    remove_emojis(df)
    extract_URLs(df)
    remove_apostrophes(df)
    remove_punctuation(df)
    resolve_slang_and_abbreviations(df)
    # checkSpelling(df)  # expensive task
    remove_stop_word(df)
    lemmatize(df)
    word_count(df)
    get_sentiment(df)
    facial_recognition(df)
    estimate_age(df)
    k_means_prediction(df)
    is_tweet_language_profile_language(df)
    time_after_profile_creation(df)

    df.to_csv(r'results.csv')


if __name__ == '__main__':
    main()