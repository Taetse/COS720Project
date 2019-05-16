import cleaning
import collections as coll

def most_common_words(df):
    all_words = []
    for row in df.head().iterrows():
        content = row['CONTENT']
        words = content.split()
        for word in words:
            all_words.append(word.lower())
    
    return coll.Counter(words).most_common(25)

def friends_followers_profile_picture(df):
    count = 0
    out = {
        "friends_face" : 0,
        "friends_no_face" : 0,
        "followers_face" : 0,
        "followers_no_face" : 0
    }
    for row in df.head().iterrows():
        count += 1
        if row['PFP_CONTAIN_FACE'] == True:
            out['friends_face'] += row['FRIENDS_COUNT']
            out['followers_face'] += row['FOLLOWERS_COUNT']
        else:
            out['friends_no_face'] += row['FRIENDS_COUNT']
            out['followers_no_face'] += row['FOLLOWERS_COUNT']

    return out

def count_non_matching_languages(df):
    out = {
        "matching" : 0,
        "non-matching" : 0
    }
    for row in df.head().iterrows():
        if row['LANGUAGE'] == row['CONTENT_LANGUAGE']:
            out['matching'] += 1
        else:
            out['non-matching'] += 1
    
    return out

def sentiment_frequency(df):
    out = {
        "positive" : 0,
        "negative" : 0,
        "no_content" : 0
    }
    for row in df.head().iterrows():
        if row['SENTIMENT'] == 1:
            out['positive'] += 1
        elif row['SENTIMENT'] == 0:
            out['no_content'] += 1
        elif row['SENTIMENT'] == -1:
            out['negative'] += 1

    return out

def sentiment_retweet_count(df):
    out = {
        "positive" : 0,
        "negative": 0
    }
    for row in df.head().iterrows():
        if row['SENTIMENT'] == 1:
            out['positive'] += row['RETWEET']
        else:
            out['negative'] += row['RETWEET']

    return out

def sentiment_word_count_distribution(df):
    print("NOT IMPLEMENTED")

def sentiment_emoji_count_distribution(df):
    print("NOT IMPLEMENTED")

def sentiment_common_word_distribution(df):
    print("NOT IMPLEMENTED")

def profile_age_follower_distribution(df):
    print("NOT IMPLEMENTED")