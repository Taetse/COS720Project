from collections import Counter
# Tweets


def most_common_words(df):
    all_words = []
    for row in df.iterrows():
        content = row[1]['CONTENT']
        if type(content) == str:
            words = content.split()
            for word in words:
                all_words.append(word.lower())

    return Counter(all_words).most_common(25)

# Tweets


def friends_followers_profile_picture(df):
    count = 0
    out = {
        "friends_face": 0,
        "friends_no_face": 0,
        "followers_face": 0,
        "followers_no_face": 0
    }
    for row in df.iterrows():
        count += 1
        if row[1]['FRIENDS'].isdigit() and row[1]['FOLLOWERS'].isdigit():
            if row[1]['PFP_CONTAIN_FACE'] == True:
                out['friends_face'] += int(row[1]['FRIENDS'])
                out['followers_face'] += int(row[1]['FOLLOWERS'])
            else:
                out['friends_no_face'] += int(row[1]['FRIENDS'])
                out['followers_no_face'] += int(row[1]['FOLLOWERS'])

    return out

# Tweets


def count_non_matching_languages(df):
    out = {
        "matching": 0,
        "non-matching": 0
    }
    for row in df.iterrows():
        if row[1]['LANGUAGE'] == row[1]['CONTENT_LANGUAGE']:
            out['matching'] += 1
        else:
            out['non-matching'] += 1

    return out

# Tweets


def sentiment_frequency(df):
    out = {
        "positive": 0,
        "negative": 0,
        "no_content": 0
    }
    for row in df.iterrows():
        if row[1]['SENTIMENT'] == 1:
            out['positive'] += 1
        elif row[1]['SENTIMENT'] == 0:
            out['no_content'] += 1
        elif row[1]['SENTIMENT'] == -1:
            out['negative'] += 1

    return out

# Tweets


def sentiment_retweet_count(df):
    out = {
        "positive": 0,
        "negative": 0
    }
    for row in df.iterrows():
        if row['SENTIMENT'] == 1:
            out['positive'] += row[1]['RETWEET']
        else:
            out['negative'] += row[1]['RETWEET']

    return out

# Tweets


def sentiment_word_count_distribution(df):
    out = {}
    for row in df.iterrows():
        if row[1]['SENTIMENT'] > 0:
            if row[1]['WORD_COUNT'] in out:
                out[row[1]['WORD_COUNT']]['positive'] += 1
            else:
                out[row[1]['WORD_COUNT']] = {}
                out[row[1]['WORD_COUNT']]['positive'] += 1
        elif row[1]['SENTIMENT'] < 0:
            if row[1]['WORD_COUNT'] in out:
                out[row[1]['WORD_COUNT']]['negative'] += 1
            else:
                out[row[1]['WORD_COUNT']] = {}
                out[row[1]['WORD_COUNT']]['negative'] += 1

    return out

# Tweets


def sentiment_emoji_count_distribution(df):
    out = {}
    for row in df.iterrows():
        if row[1]['SENTIMENT'] > 0:
            if row[1]['EMOJI_COUNT'] in out:
                out[row[1]['EMOJI_COUNT']]['positive'] += 1
            else:
                out[row[1]['EMOJI_COUNT']] = {}
                out[row[1]['EMOJI_COUNT']]['positive'] += 1
        elif row[1]['SENTIMENT'] < 0:
            if row[1]['EMOJI_COUNT'] in out:
                out[row[1]['EMOJI_COUNT']]['negative'] += 1
            else:
                out[row[1]['EMOJI_COUNT']] = {}
                out[row[1]['EMOJI_COUNT']]['negative'] += 1

    return out


def emoji_count_distribution(df):
    out = {}
    for row in df.iterrows():
        if row[1]['EMOJI_COUNT'] in out:
            out[row[1]['EMOJI_COUNT']] += 1
        else:
            out[row[1]['EMOJI_COUNT']] = 1
            # out[row[1]['EMOJI_COUNT']] += 1

    return out
# Tweets


def count_phishing(df):
    out = {'phishing': 0, 'legit': 0}
    for row in df.iterrows():
        if row[1]['CONTAINS_PHISHING']:
            out['phishing'] += 1
        else:
            out['legit'] += 1

    return out


def sentiment_common_word_distribution(df, words):
    out = {}
    for row in df.iterrows():
        content = row[1]['CONTENT']
        for word in words:
            if word in content:
                if row[1]['SENTIMENT'] > 0:
                    if word in out:
                        out[word]['positive'] += 1
                    else:
                        out[word] = {}
                        out[word]['positive'] += 1
                elif row[1]['SENTIMENT'] < 0:
                    if word in out:
                        out[word]['negative'] += 1
                    else:
                        out[word] = {}
                        out[word]['negative'] += 1

    return out

# Tweets


def profile_age_follower_distribution(df):
    def get_profile_age(df):
        from dateutil import parser
        import datetime

        def get_age(dateCreated):
            return today.year - dateCreated.year - ((today.month, today.day) < (dateCreated.month, dateCreated.day))
        today = datetime.date.today()
        df['PROFILE_AGE'] = df['CREATED'].apply(
            lambda x: get_age(parser.parse(x))
        )
        print(df.head()[['CREATED', "PROFILE_AGE"]])
    print('-------Follower count--------')
    get_profile_age(df)
    temp = {
        5: 0,
        10: 0,
        15: 0,
    }
    for row in df.iterrows():
        key = 15
        age = row[1][27]
        if age <= 5:
            key = 5
        elif age <= 10:
            key = 10
        temp[key] += row[1][12]
    print(temp)
    return temp
