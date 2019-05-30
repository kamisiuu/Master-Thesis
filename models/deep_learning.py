from datetime import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
#https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
#https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-10-neural-network-with-a6441269aa3c
# https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/


from keras import preprocessing, utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("../datasets/dataset_1/train.csv", header='infer', index_col=None)
x_train, x_test, y_train, y_test = train_test_split(train["SentimentText"], train["Sentiment"], random_state=1000,
                                                    test_size=0.3)


# nb_classes= np.max(y_train) + 1
# from keras.utils import np_utils
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# >>> COUNT VECTORIZER >>>
count_vect = CountVectorizer()
X=count_vect.fit_transform(x_train)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# >>> TUNING BATCH SIZE AND EPOCHS >>>
def create_model():
    # Build the model
    model = Sequential()
    input_dim = X.shape[1]
    model.add(Dense(512, input_dim=input_dim))
    model.add(Dense(num_classes))
    model.add(Dropout(0.1))
    model.add(Activation('softmax'))

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=['accuracy'])
    return model

start=time()
model = KerasClassifier(build_fn=create_model)
optimizer = ['rmsprop', 'adam']
epochs = (50, 100, 150)
batches = (5, 10, 20)
param_grid = dict(epochs=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid)
grid_result = grid.fit(X,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
# print("total time:",time()-start)

# <<< TUNING BATCH SIZE AND EPOCHS <<<

# >>> TUNING


# batch_size = 512
# epochs = 40

# #Build the model
# model = Sequential()
# model.add(Dense(512, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
# score = model.evaluate(x_test, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test accuracy:', score[1])
# build lstm
# https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47

# class LSTM:
#     def __init__(self,x_train,y_train,x_test,y_test,embed_dim, lstm_out, batch_size, epochs,):
#         self.x_train,self.y_train,self.x_test,self.y_test=x_train,y_train,x_test,y_test
#         self.embed_dim= embed_dim
#         self.lstm_out = lstm_out
#         self.batch_size = batch_size
#         self.epochs = epochs
#
#     def run(self):
#         model = Sequential()
#         model.add(Embedding(2500, self.embed_dim, input_length=self.x_train.shape[1], dropout=0.2))
#         model.add(LSTM(self.lstm_out, dropout_U=0.2, dropout_W=0.2))
#         model.add(Dense(2, activation='softmax'))
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         print(model.summary())
#         model.fit(self.x_train,self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
#         score = model.evaluate(self.x_test, self.y_test,
#                                batch_size=self.batch_size, verbose=1)
#         print('Test accuracy:', score[1])
#         return model
