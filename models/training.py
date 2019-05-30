#
# AUTHOR: KAMIL LIPSKI
#

# >>> IMPORTS >>>
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics, preprocessing
from xgboost.sklearn import XGBClassifier
from tensorflow import keras
from keras.layers import Embedding, LSTM, Dense
from keras import models as mlp
from models.data_exploring import ExploringData
from models import tweet_cleaner as dataclean
from models.grid_search_utility import grid_search_svc
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils
from keras.models import Sequential
from keras import layers, optimizers, Model
import time

#<<< IMPORTS <<<

def Train(train,datasetname,train_tweet,train_label, dataexplore=False, storemodel=False):
    start = time.time()

    if dataexplore:
        exp1= ExploringData(train,train_tweet,train_label)
        exp1.runall()
    # >>> PREPROCESSING START >>>
    #train = dataclean.tweet_cleaner(train, train_tweet,
    #                                preprocessoptions=['noise', 'short_words', 'stop_words', 'rare_words',
    #                                                  'common_words', 'tokenization', 'lower_case'])
    # splitting data into training and validation set
    x_train, x_test, y_train, y_test = train_test_split(train[train_tweet], train[train_label], random_state=1000,
                                                      test_size=0.3)


    # >>> COUNT VECTORIZER >>>
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=2500)
    count_vect.fit(train[train_tweet])
    xtrain_count = count_vect.transform(x_train)
    xvalid_count = count_vect.transform(x_test)
    # <<< COUNT VECTORIZER <<<

    # >>> TF-IDF WORD LEVEL >>>
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=2500)
    tfidf_vect.fit(train[train_tweet])
    xtrain_tfidf = tfidf_vect.transform(x_train)
    xvalid_tfidf = tfidf_vect.transform(x_test)
    # <<< TF-IDF WORD LEVEL <<<

    # >>> TF-IDF NGRAM-LEVEL >>>
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                       max_features=2500)
    tfidf_vect_ngram.fit(train[train_tweet])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(x_train)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(x_test)
    # <<< TF-IDF NGRAM-LEVEL <<<

    # >>> TF-IDF CHARACTER LEVEL >>>
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                             max_features=2500)
    tfidf_vect_ngram_chars.fit(train[train_tweet])
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(x_train)
    xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(x_test)
    # <<< TF-IDF CHARACTER LEVEL <<<

    # >>> BAG OF WORDS >>>
    train_bow = CountVectorizer(max_features=2500, lowercase=True, ngram_range=(1, 3), analyzer="word")
    train_bow.fit(train[train_tweet])
    xtrain_bow = train_bow.transform(x_train)
    xvalid_bow = train_bow.transform(x_test)
    # <<< BAG OF WORDS <<<

    class Feature:
        def __init__(self, name, xtrain=[], xvalid=[]):
            self.name = name
            self.xtrain = xtrain
            self.xvalid = xvalid

    featureList = []
    featureList.append(Feature('BAG-OF-WORDS', xtrain_count, xvalid_count))
    featureList.append(Feature('BAG-OF-WORDS-NGRAM', xtrain_bow, xvalid_bow))
    featureList.append(Feature('TF-IDF-WORD', xtrain_tfidf, xvalid_tfidf))
    featureList.append(Feature('TF-IDF-NGRAM-WORD', xtrain_tfidf_ngram, xvalid_tfidf_ngram))
    featureList.append(Feature('TF-IDF-NGRAM-CHARS', xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars))
    # <<< PREPROCESSING END <<<

    def train_model(classifier, model_name, feature_name, feature_vector_train, label, feature_vector_valid, neural_net=False, batch_size=False):
        # fit the training dataset on the classifier
        model = classifier
        model.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = model.predict(feature_vector_valid)
        entries.append((datasetname, model_name, metrics.accuracy_score(predictions, y_test), feature_name))
        return metrics.accuracy_score(predictions, y_test)



    def neural_network(batch_size,epochs,dropout,optimizer,loss,feature_vector_train,feature_vector_valid,feature_name):

        # Build the model
        model = Sequential()
        input_dim= feature_vector_train.shape[1]
        model.add(Dense(512, input_dim=input_dim))

        model.add(Dropout(dropout))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss=loss,optimizer=optimizer,
                      metrics=['accuracy'])

        history = model.fit(feature_vector_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1
                            )
        score = model.evaluate(feature_vector_valid, y_test,
                               batch_size=batch_size, verbose=1)
        storeresults.append((datasetname, "Deep Neural Networks",score[1], feature_name, epochs, batch_size, droput,optimizer,loss))
        print('Test accuracy:', score[1])

        return model

    def neural_network_grid_search(feature_vector_train, epochs_and_batch):
        def create_model():
            # Build the model
            model = Sequential()
            input_dim = feature_vector_train.shape[1]
            model.add(Dense(512, input_dim=input_dim))
            model.add(Dense(num_classes))
            model.add(Dropout(0.1))
            model.add(Activation('softmax'))

            model.compile(loss="binary_crossentropy", optimizer="adam",
                          metrics=['accuracy'])
            return model
        if epochs_and_batch:

            model = KerasClassifier(build_fn=create_model)
            optimizer = ['rmsprop', 'adam']
            epochs = (50, 100, 150)
            batches = (5, 10, 20)
            param_grid = dict(epochs=epochs, batch_size=batches)
            grid = GridSearchCV(estimator=model,
                                param_grid=param_grid)
            grid_result = grid.fit(feature_vector_train, y_train)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))







    class Classifier:
        def __init__(self,clname, classifiermodel):
            self.clname=clname
            self.classifiermodel=classifiermodel

    # >>> CLASSIFIERS >>>
    entries = []
    models = [MultinomialNB(), LogisticRegression(), SGDClassifier(), KNeighborsClassifier(), RandomForestClassifier(),
              XGBClassifier(), MLPClassifier()]

    classifierList = []
    classifierList.append(Classifier('Keras', 'create_keras()'))
    # <<< CLASSIFIERS <<<

    # >>> SVC GRID SEARCH >>>
    #[grid_search_svc(feature.xtrain,y_train) for feature in featureList]
    #grid_search_svc(xtrain_count,y_train)
    # <<< SVC GRID SEARCH <<<




    # >>> TRADITIONAL MACHINE LEARNING METHODS >>>
    # [train_model(model,model.__class__.__name__, feature.name, feature.xtrain, y_train, feature.xvalid)for model in
    # models for feature in featureList]
    # <<< TRADITIONAL MACHINE LEARNING METHODS <<<


    # >>> DEEP LEARNING >>>
    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    batch_size=512
    epochs=40
    droput=0.5
    optimizer="adam"
    loss="binary_crossentropy"

    storeresults=[]
    [neural_network(batch_size=batch_size, epochs=epochs, dropout=droput, optimizer=optimizer, loss=loss, feature_vector_train=feature.xtrain,
                    feature_vector_valid=feature.xvalid,feature_name=feature.name) for feature in featureList]
    # <<< DEEP LEARNING <<<


    # >>> GRID SEARCH DEEP LEARRNING >>>
    #neural_network_grid_search(xtrain_count,epochs_and_batch=True)
    # <<< GRID SEARCH DEEP LEARNING >>>

    # >>> WRITE RESULTS >>>
    cv_tm = pd.DataFrame(entries,columns=['dataset','model_name', 'accuracy','feature'])
    cv_at = pd.DataFrame(storeresults, columns=['dataset','model_name','accuracy','feature','epochs', 'batch_size', 'droput','optimizer','loss'])
    # print(cv_df)
    cv_tm.to_csv('results/accuracy_table/all_results_from_traditional_training.csv',mode='a', index_label=False, header=False,
                 index=False)
    cv_at.to_csv('results/accuracy_table/all_results_from_deep_learning.csv', mode='a', index_label=False, header=False,
                 index=False)
    # <<< WRITE RESULTS <<<


    end = time.time()
    result = (end - start) / 60
    print("time to train "+datasetname+" took: ", str(result), " minutes")