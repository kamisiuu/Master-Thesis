import pandas as pd
from classes import data_exploring
from classes.data_exploring import ExploringData
from classes.data_upload import Upload
#train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
#test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)

features = [[1,2],[3,4]]

for feature in features:
    print(feature[0])


