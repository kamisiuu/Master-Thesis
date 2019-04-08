import pandas as pd
from classes import data_exploring
from classes.data_exploring import ExploringData
from classes.data_upload import Upload
train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
#test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)

from classes.tweet_cleaner import tweet_cleaner
#data = pd.read_csv('data/results/accuracy_table/all_results_from_training.csv')
print(train)

data = tweet_cleaner(train,'tweet',preprocessoptions=['noise','short_words'])
print(data)