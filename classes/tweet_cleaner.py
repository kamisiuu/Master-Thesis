import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import *

tokenized_tweet = []



def noise(data,tweetcolumn):
    """ @Author = Kamil Lipski #
    noise () function  removes special characters, numbers, punctuations and removes twitter handles (@user)

    :param data: you have to give it a dataset
    :param tweetcolumn: you have to specify the column name that you what to remove noise from
    :return: it returns the preprocessed dataset
    """
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt
    data[tweetcolumn] = data[tweetcolumn].str.replace("[^a-zA-Z#]", " ")
    data[tweetcolumn] = np.vectorize(remove_pattern)(data[tweetcolumn], "@[\w]*")
    return data



def short_words(data,tweetcolumn):
    """ @Author = Kamil Lipski #
    short_words() function removes all short words from dataset

    :param data: you have to give it a dataset
    :param tweetcolumn: you have to specify the column name that you what to remove noise from
    :return: it returns the preprocessed dataset
    """
    data[tweetcolumn] = data[tweetcolumn].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    return data



def stop_words(data,tweetcolumn):
    """@Author = Kamil Lipski #
    stop_words () function removes commonly occurring words from the text data.

    :param data: you have to give it a dataset
    :param tweetcolumn: you have to specify the column name that you what to remove noise from
    :return: it returns the preprocessed dataset
    """
    stop = stopwords.words('english')
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return data



def rare_words(data,tweetcolumn):
    """@Author = Kamil Lipski #
    rare_words () function removes rare occurring words from the text data.

    :param data: you have to give it a dataset
    :param tweetcolumn: you have to specify the column name that you what to remove noise from
    :return: it returns the preprocessed dataset
    """
    freq = pd.Series(' '.join(data[tweetcolumn]).split()).value_counts()[-10:]
    freq = list(freq.index)
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    return data



def lower_case(data,tweetcolumn):
    """@Author = Kamil Lipski #
    lower_case () function transoforms words from the text data to lower case.

    :param data: you have to give it a dataset
    :param tweetcolumn: you have to specify the column name that you what to remove noise from
    :return: it returns the preprocessed dataset
    """
    data[tweetcolumn] = data[tweetcolumn].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return data
def datareturn(data):
    return data
class tweet_cleaner:
    """This is a constructor of class tweet_cleaner
        @Author=Kamil Lipski
    You create this class in following way:
       data = tweet_cleaner(dataset,columnwithtweets,cleaningoptions=['stop_words','rare_words'])
    """
    def __new__(self,data,tweetcolumn, preprocessoptions=[]):
        """
               @Author=Kamil Lipski # This is a class constructor of class tweet_cleaner

               :param data: you have to give it a dataset
               :param tweetcolumn: you have to specify the column name that you what to remove noise from
               :param preprocessoptions: 'noise'-removal of special characters,numbers,punctutation, twitter handles # 'short_words'-removal of short words #'stop_words'- removal of stop words #'rare_words'- removal of rare words#'lower_case' - transforms words into lower case #'tokenization' - transforms text into vectors #'stemming' - stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.
               :return: returns dataset after preprocessing
               """
        self.data = data
        self.tweetcolumn = tweetcolumn
        self.preprocessoptions = preprocessoptions  # this gives
        choiceList = {'noise': noise(data, tweetcolumn),
                      'short_words': short_words(data, tweetcolumn),
                      'stop_words': stop_words(data, tweetcolumn),
                      'rare_words': rare_words(data, tweetcolumn),
                      'lower_case': lower_case(data, tweetcolumn)}
        for option in preprocessoptions:
            self.data = choiceList.get(option)
        return self.data



def clean_data(train,train_tweet):
    # this function helps to remove twitter handles , twitter data contain lots of twitter handles like "@username",
    # we remove those signs and usernames, because we dont won't to store peoples information, and at the same time this type of information
    # is not needed for our tests










    # Common word removal
    freq = pd.Series(' '.join(train[train_tweet]).split()).value_counts()[:10]
    freq = list(freq.index)
    train[train_tweet] = train[train_tweet].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # # Spelling correction
    # from textblob import TextBlob
    # train[train_tweet].apply(lambda x: str(TextBlob(x).correct()))

    # Tokenization process of building a dictionary and transform document into vectors
    tokenized_tweet = train[train_tweet].apply(lambda x: x.split())


    # Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc)
    # from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the
    # different variations of the word – “play”.

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming


    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    train[train_tweet] = tokenized_tweet
    return train





