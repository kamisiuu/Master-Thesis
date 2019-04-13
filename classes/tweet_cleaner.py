import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import *
from textblob import Word


def noise(dataset, tweetcolumn):
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    dataset[tweetcolumn] = dataset[tweetcolumn].str.replace("[^a-zA-Z#]", " ")
    dataset[tweetcolumn] = np.vectorize(remove_pattern)(dataset[tweetcolumn], "@[\w]*")
    return dataset


def short_words(dataset, tweetcolumn):
    dataset[tweetcolumn] = dataset[tweetcolumn].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    return dataset


def stop_words(dataset, tweetcolumn):
    stop = stopwords.words('english')
    dataset[tweetcolumn] = dataset[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    print("stop words")
    return dataset


def rare_words(dataset, tweetcolumn):
    freq = pd.Series(' '.join(dataset[tweetcolumn]).split()).value_counts()[-10:]
    freq = list(freq.index)
    dataset[tweetcolumn] = dataset[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return dataset


def common_words(dataset, tweetcolumn):
    freq = pd.Series(' '.join(dataset[tweetcolumn]).split()).value_counts()[:10]
    freq = list(freq.index)
    dataset[tweetcolumn] = dataset[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return dataset


def tokenization(dataset, tweetcolumn):
    dataset = dataset[tweetcolumn].apply(lambda x: x.split())
    return dataset


def stemming(dataset, tweetcolumn):
    stemmer = PorterStemmer()
    tokenized_tweet = tokenization(dataset, tweetcolumn)
    tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    return dataset


def lemmatization(dataset, tweetcolumn):
    dataset[tweetcolumn] = dataset[tweetcolumn].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return dataset


def lower_case(dataset, tweetcolumn):
    dataset[tweetcolumn] = dataset[tweetcolumn].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return dataset

class tweet_cleaner:
    global choiceList,dataset
    choiceList = {'noise': 'noise(dataset, tweetcolumn)',
                  'short_words': 'short_words(dataset, tweetcolumn)',
                  'stop_words': 'stop_words(dataset, tweetcolumn)',
                  'rare_words': 'rare_words(dataset, tweetcolumn)',
                  'common_words': 'common_words(dataset, tweetcolumn)',
                  'tokenization': 'tokenization(dataset, tweetcolumn)',
                  'stemming': 'stemming(dataset, tweetcolumn)',
                  'lemmatization': 'lemmatization(dataset, tweetcolumn)',
                  'lower_case': 'lower_case(dataset, tweetcolumn)'}
    def __new__(cls, dataset, tweetcolumn, preprocessoptions=False):
        """
        :param dataset: give it an DataFrame dataset
        :param tweetcolumn: the column you want to preprocess
        :param preprocessoptions: if not defined all cleaning options run automatically else if you define them you do it in following way
                preprocessoptions=['noise','short_words','stop_words','rare_words','common_words','stemming','lemmatization','lower_case']
                you can choose only between some of them too
        :return: return the dataset
        """
        cls.dataset = dataset
        cls.tweetcolumn= tweetcolumn
        cls.preprocessoptions = preprocessoptions

        cls.dataset = [exec(choiceList.get(cls, option)) for option in cls.preprocessoptions if preprocessoptions]
        cls.dataset = [exec(choiceList[choice]) for choice in choiceList if not preprocessoptions]

        return dataset
