import pandas as pd
from classes import data_exploring
from classes.data_exploring import ExploringData
from classes.data_upload import Upload
train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
#test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)

from classes.tweet_cleaner import tweet_cleaner
#data = pd.read_csv('data/results/accuracy_table/all_results_from_training.csv')


data = tweet_cleaner(train,'tweet',preprocessoptions=['noise'])
print(data)


# global listen
# listen= [1,2,3,4]
# def fun(variable):
#     listen.remove(variable)
#
# class Test:
#     global choicesList
#     choicesList = {'a': 'fun(2)',
#             'b':'fun(1)',
#             'c':'fun(3)'}
#     def __new__(cls,arry=[]):
#         cls.arry=arry
#
#         for element in cls.arry:
#             mycode = choicesList.get(element)
#             exec(mycode)
#
# tt= Test(arry=['a','b'])
# print(listen)






