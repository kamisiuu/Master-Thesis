import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import *

tokenized_tweet = []
def clean_data(train,train_tweet):
    # this function helps to remove twitter handles , twitter data contain lots of twitter handles like "@username",
    # we remove those signs and usernames, because we dont won't to store peoples information, and at the same time this type of information
    # is not needed for our tests
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)

        return input_txt


    # remove twitter handles (@user)
    train[train_tweet] = np.vectorize(remove_pattern)(train[train_tweet], "@[\w]*")
    # remove special characters, numbers, punctuations
    train[train_tweet] = train[train_tweet].str.replace("[^a-zA-Z#]", " ")
    # removing short words
    train[train_tweet] = train[train_tweet].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    # transforming into lower case
    train[train_tweet] = train[train_tweet].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Removal of stopwords

    stop = stopwords.words('english')
    train[train_tweet] = train[train_tweet].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Common word removal
    # freq = pd.Series(' '.join(combi['tidy_tweet']).split()).value_counts()[:10]
    # freq = list(freq.index)
    # combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # Rare word removal
    # freq = pd.Series(' '.join(combi['tidy_tweet']).split()).value_counts()[-10:]
    # freq = list(freq.index)
    # combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


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





