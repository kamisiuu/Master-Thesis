#
# AUTHOR: KAMIL LIPSKI
#
import gensim
import pandas as pd
from models.training import Train
from models.data_exploring import ExploringData
# resources:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://cmdlinetips.com/2018/11/string-manipulations-in-pandas/
# w3schools.com/python
# https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

train_1 = pd.read_csv("datasets/dataset_1/train.csv", header='infer', index_col=None)
train_2 = pd.read_csv("datasets/dataset_2/train.csv", header='infer',index_col=None)
train_3 = pd.read_csv("datasets/dataset_3/train.csv", delimiter=None, header='infer', names=None,  encoding='latin-1', error_bad_lines=False)
train_4 = pd.read_csv("datasets/dataset_4/train.csv", delimiter=None, header='infer', names=None,  encoding='latin-1')

train_3=train_3[:500000]

Train(train_1,"dataset_1","SentimentText","Sentiment")
#Train(train_2,'dataset_2',"SentimentText","Sentiment")
#Train(train_3,'dataset_3',"SentimentText","Sentiment")
#Train(train_4,'dataset_4',"SentimentText","Sentiment")
exit(0)