#
# AUTHOR: KAMIL LIPSKI
#
import gensim
import pandas as pd
from classes.training import Train

# resources:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://cmdlinetips.com/2018/11/string-manipulations-in-pandas/
# w3schools.com/python
# https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

train_1 = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
test_1 = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)
train_2 = pd.read_csv("data/dataset_2/train.csv", header='infer',index_col=None)
test_2 = pd.read_csv("data/dataset_2/test.csv", delimiter=None, header='infer', names=None, index_col=None, encoding='latin-1')

#Train(train_1,"dataset_1","tweet","label")
Train(train_2,'dataset_2',"SentimentText","Sentiment")

exit(0)