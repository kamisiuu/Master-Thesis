import pandas as pd
from classes import data_exploring
from classes.data_exploring import ExploringData
from classes.data_upload import Upload
#train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
#test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)

train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
test = pd.read_csv("data/dataset_1/test.csv", delimiter=None, header='infer', names=None, index_col=None, encoding='latin-1')

print(train.shape)
print(test.shape)
ex1=ExploringData(train,"tweet","label")

#ex1=ExploringData(train,"SentimentText","Sentiment")
ex1.runall()


