import pandas as pd
import csv
import re
from textblob import TextBlob

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


def read_from_csv(file_name: str):
    df = pd.read_csv(file_name, sep=',', quotechar='\"', escapechar='\\', error_bad_lines=False, skipinitialspace=True)
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
    file_name = "D:/Documents/COS 720/COS720Project/data-cleaning/slang.txt"
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
        lambda x: ' '.join([lmtzr.lemmatize(word, 'v') for word in x.split() ]))
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


def get_sentiment(df):
    df['SENTIMENT'] = df['CONTENT'].apply(
        lambda x: TextBlob(x).sentiment.polarity)

    print('-------Sentiment Analysis--------')
    print(df.head()[['CONTENT', "SENTIMENT"]])


def main():
    df = read_from_csv(r"D:\Documents\COS 720\shortened\EX\EXP_TWEETS_DETAIL\shortened-data.csv")

    print("--- Print the Head of the data ---")
    print(df.head()["CONTENT"])

    # detect_language(df)
    remove_mentions(df)
    count_emojis(df)
    remove_emojis(df)
    resolve_slang_and_abbreviations(df)
    remove_stop_word(df)
    lemmatize(df)
    to_lower(df)
    get_sentiment(df)


if __name__ == '__main__':
    main()
