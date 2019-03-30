from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

class Upload:

    def __init__(self, path, separator):
        self.path=path
        self.separator=separator
        labels, texts = [], []
        data = open(self.path).read()
        for i, line in enumerate(data.split(self.separator)):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
        trainDF = pandas.DataFrame()
        trainDF['text'] = texts
        trainDF['label'] = labels
        return trainDF



