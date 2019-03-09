import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import pprint
import warnings
from sklearn import svm
from tabulate import tabulate
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.stem.porter import *

# resources:
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
# https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-dataset-python/
# http://cmdlinetips.com/2018/11/string-manipulations-in-pandas/
# https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/

# importing data
print("Uploading train and test")
train = pd.read_csv("../data/training.txt", sep="\t", header=None)
test = pd.read_csv("../data/testdata.txt", delim_whitespace=False, delimiter="\n", header=None)
train.columns = ["label", "tweet"]
test.columns = ["tweet"]

# show the dimensions of the dataset, first value shows how many instances(rows) we have in each datasets,
# and the seconds says about the number of attributes(columns)
#print("uploading successful")
#print("Dimensions of the train, and : ", train.shape)
#print("Dimensions of the test: ", test.shape)

#statistical summary
#print("statistical summary of train")
#print(train.describe())
#print("statistical summary of test")
#print(test.describe())

# print out the head of the data, this shows a first 20 rows of each uploaded dataset,test and train
#print(train.head(20))
#print(test.head(20))

# combining train and test, this saves the trouble of performing the same steps twice on test and train
combi = train.append(test, ignore_index=False, sort=False)

# print(combi)

# this function helps to remove twitter handles , twitter data contain lots of twitter handles like "@username", and other unecessary repetitions of characters or punctuations
# we remove those signs and usernames, because we dont won't to store peoples information, and at the same time this type of information
# is not needed for our tests
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# Basic PreProcessing
# cleaned data will be stored in another list and named as combi containing tidy tweets
# remove twitter handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
# remove special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
# removing short words
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# Tokenization
# tokenization-splitting or into separate words,

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

# tokenization-splitting into separate words,

# it is a process of splitting whole tweets(sentences) into tokens(separate words)
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())



# Stemming. Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) 
# from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the 
# different variations of the word – “play”.
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])


for i in range(len(tokenized_tweet)):
    s = ""
for j in tokenized_tweet.iloc[i]:
    s += "".join(j) + ""
tokenized_tweet.iloc[i] = s.rstrip()

all_words = ' '.join([text for text in combi['tidy_tweet']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

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

tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


train_bow = bow[:6918,:]
test_bow = bow[6918:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
#print(f1_score(yvalid, prediction_int)) # calculating f1 score
print(tabulate([["Bag-of-words","logistic regression",f1_score(yvalid, prediction_int)]], headers=["Building model with","algorithm","accuracy"]))

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['label']]
submission.to_csv('Results_from_LREG_labeled_test.csv', index=False) # writing data to a CSV file

train_tfidf = tfidf[:6918,:]
test_tfidf = tfidf[6918:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

logregresult=lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

print(tabulate([["TF-IDF features","logistic regression",f1_score(yvalid, prediction_int)]], headers=["Building model with","algorithm","accuracy"]))
#print(f1_score(yvalid, prediction_int))

print('\n')
print("Results! Predicted labeled test dataset","\n",test.head(10))

exit(0)