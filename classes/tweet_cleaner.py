import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import *
from textblob import Word


def noise(data, tweetcolumn):
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    data[tweetcolumn] = data[tweetcolumn].str.replace("[^a-zA-Z#]", " ")
    data[tweetcolumn] = np.vectorize(remove_pattern)(data[tweetcolumn], "@[\w]*")
    return data


def short_words(data, tweetcolumn):
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    return data


def stop_words(data, tweetcolumn):
    stop = stopwords.words('english')
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    print("stop words")
    return data


def rare_words(data, tweetcolumn):
    freq = pd.Series(' '.join(data[tweetcolumn]).split()).value_counts()[-10:]
    freq = list(freq.index)
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return data


def common_words(data, tweetcolumn):
    freq = pd.Series(' '.join(data[tweetcolumn]).split()).value_counts()[:10]
    freq = list(freq.index)
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return data


def tokenization(data, tweetcolumn):
    data = data[tweetcolumn].apply(lambda x: x.split())
    return data


def stemming(data, tweetcolumn):
    stemmer = PorterStemmer()
    tokenized_tweet = tokenization(data, tweetcolumn)
    tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    return data


def lemmatization(data, tweetcolumn):
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return data


def lower_case(data, tweetcolumn):
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return data


class tweet_cleaner:
    global choiceList,data
    choiceList = {'noise': 'noise(data, tweetcolumn)',
                  'short_words': 'short_words(data, tweetcolumn)',
                  'stop_words': 'stop_words(data, tweetcolumn)',
                  'rare_words': 'rare_words(data, tweetcolumn)',
                  'common_words': 'common_words(data, tweetcolumn)',
                  'tokenization': 'tokenization(data, tweetcolumn)',
                  'stemming': 'stemming(data, tweetcolumn)',
                  'lemmatization': 'lemmatization(data, tweetcolumn)',
                  'lower_case': 'lower_case(data, tweetcolumn)'}
    def __new__(cls, data, tweetcolumn, preprocessoptions=[]):
        """@Author=Kamil Lipski # This is a class constructor of class tweet_cleaner
        preprocessoptions = ['noise','short_words','stop_words','rare_words','common_words','stemming','lemmatization','lower_case']
        :param data: you have to give it a dataset
        :param tweetcolumn: you have to specify the column name that you what to remove noise from
        :param preprocessoptions: noise: -removal of special characters,numbers,punctutation, twitter handles # 'short_words'-removal of short words #'stop_words'- removal of stop words #'rare_words'- removal of rare words#'lower_case' - transforms words into lower case #'tokenization' - transforms text into vectors #'stemming' - stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.

        :return: returns dataset after preprocessing
        """

        cls.data = data
        cls.tweetcolumn= tweetcolumn
        cls.preprocessoptions = preprocessoptions  # this gives

        for option in cls.preprocessoptions:
            mycode= choiceList.get(option)
            cls.data = exec(mycode)
        mycode=0
        return data













