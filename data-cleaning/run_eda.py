import eda
import csv

import cv2 as cv
import numpy as np
import pandas as pd

def read_from_csv(file_name: str):
    df = pd.read_csv(file_name, encoding='utf-8', sep=',', quotechar='\"', escapechar='\\', error_bad_lines=False, skipinitialspace=True)
    return df

def main():
    # df = pd.read_csv(r"D:\UP\COS\720\repo\results.csv")
    df = read_from_csv(r"D:\UP\COS\720\repo\results.csv")
    # commonWords = eda.most_common_words(df)
    # print(commonWords)
    # print(eda.emoji_count_distribution(df))
    # print(eda.friends_followers_profile_picture(df))
    # print(eda.count_non_matching_languages(df))
    # print(eda.sentiment_frequency(df))
    # print(eda.sentiment_retweet_count(df))
    # print(eda.sentiment_word_count_distribution(df))
    # print(eda.sentiment_emoji_count_distribution(df))

    # words = []
    # for w in commonWords:
    #     words.append(w[0])

    # print(eda.sentiment_common_word_distribution(df, words))
    print(eda.profile_age_follower_distribution(df))


if __name__ == '__main__':
    main()
