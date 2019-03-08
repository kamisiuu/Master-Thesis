# important notice!
# stemming cuzes problems like given a word loving it removes ing at the end and we are again with lov where love and lov are actually the same words but the program separates them


#nltk.download()
import re
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd, numpy as np
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
from nltk.stem.porter import *


# resources:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://cmdlinetips.com/2018/11/string-manipulations-in-pandas/
#https://github.com/prateekjoshi565/twitter_sentiment_analysis/blob/master/code_sentiment_analysis.ipynb

# importing data
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate

train = pd.read_csv("../data/training.txt", sep="\t", header=None)
test = pd.read_csv("../data/testdata.txt", delim_whitespace=False, delimiter="\n", header=None)
train.columns = ["label", "tweet"]
test.columns = ["tweet"]

# combining train and test together just to spare some time later
combi = train.append(test, ignore_index=False, sort=False)

# this function helps to remove twitter handles , twitter data contain lots of twitter handles like "@username",
# we remove those signs and usernames, because we dont won't to store peoples information, and at the same time this type of information
# is not needed for our tests
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# cleaned data will be stored in another list and named as combi containing tidy tweets
# remove twitter handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
# remove special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
# removing short words
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
#transforming into lower case
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Removal of stopwords
from nltk.corpus import stopwords
#nltk.download('stopwords')
stop = stopwords.words('english')
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Common word removal
#freq = pd.Series(' '.join(combi['tidy_tweet']).split()).value_counts()[:10]
#freq = list(freq.index)
#combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#Rare word removal
#freq = pd.Series(' '.join(combi['tidy_tweet']).split()).value_counts()[-10:]
#freq = list(freq.index)
#combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

#Spelling correction
from textblob import TextBlob
combi['tidy_tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

# Tokenization
# tokenization is a process of building a dictionary and transform document into vectors
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())


# Stemming.
# Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc)
# from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the
# different variations of the word – “play”.
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
for i in range(len(tokenized_tweet)):
    s = ""
for j in tokenized_tweet.iloc[i]:
    s += "".join(j) + ""
tokenized_tweet.iloc[i] = s.rstrip()


# Bag of words feautures
# Bag-of-Words is a method to represent text into numerical features.
# Consider a corpus (a collection of texts) called C of D documents
# {d1,d2…..dD} and N unique tokens extracted out of the corpus C.
# The N tokens (words) will form a list, and the size of the bag-of-words
# matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])


# TF-IDF feature matrix
# This is another method which is based on the frequency method
# but it is different to the bag-of-words approach
# in the sense that it takes into account,
# not just the occurrence of a word in a single document (or tweet)
# but in the entire corpus.
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


# Model building: Sentiment Analysis
# We are now done with all the pre-modeling stages required
# to get the data in the proper form and shape. Now we will
# be building predictive models on the dataset using the two
# feature set — Bag-of-Words and TF-IDF.
train_bow = bow[:6918,:]
test_bow = bow[6918:,:]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(xtrain_bow,ytrain)
prediction_int = clf.predict(xvalid_bow)
print(tabulate([["Multilayer bag-of-words","Neural Networks",f1_score(yvalid, prediction_int)]], headers=["Building model with","algorithm","accuracy"]))

# Building model using TF-IDF features
train_tfidf = tfidf[:6918,:]
test_tfidf = tfidf[6918:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]


#when we change number of hidden layers, algorithm gives poorer results
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1)

logregresult=clf.fit(xtrain_tfidf, ytrain)

prediction = clf.predict(xvalid_tfidf)

print('\n')
#print(f1_score(yvalid, prediction_int)) # calculating f1 score
print(tabulate([["Multilayer TF-IDF features","Neural Networks",f1_score(yvalid, prediction_int)]], headers=["Building model with","algorithm","accuracy"]))
# We trained the support vector machine model on the Bag-of-Words features and it gave us an F1-score.
# Now we will use this model to predict for the test data.
test_pred = clf.predict(test_bow)
print('\n')
print(tabulate([["on test set","Neural Networks",f1_score(yvalid, prediction_int)]], headers=["Prediction and Evaluation","algorithm","accuracy"]))

test['label'] = test_pred
submission = test[['tweet','label']]
col_names = ['tweet','label']
submission.to_csv('Results_from_MLP_NN_labeled_test.csv', index=False, index_label=None) # writing data to a CSV file
print('\n')
print("Results! Predicted labeled test dataset","\n",test.head(10))

# using this last function you can write what ever sentence you want to , to determine polarity
sentence_to_predict = "i'm loving shanghai"
print('predicting if sentence: "', sentence_to_predict, '" is positive or negative')
print(clf.predict(tfidf_vectorizer.transform([sentence_to_predict])))


def sentence_predictor(sentence):

    prediction_sentence = clf.predict(tfidf_vectorizer.transform([sentence]))
    if (prediction_sentence == 0):
           result = "Computer calculated that sentence is negative"
    else:
        result = "Computer calculated that sentence is positive"
    return result

print(sentence_predictor(sentence_to_predict))
exit(0)