import pandas as pd
from classes import data_exploring
from classes.data_exploring import ExploringData

train = pd.read_csv("data/dataset_2/train.csv", header='infer', index_col=None)

ex1=ExploringData(train,"SentimentText","Sentiment")

ex1.runall()