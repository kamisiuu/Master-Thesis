#
# AUTHOR: KAMIL LIPSKI
#

# >>> IMPORTS >>>
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics, preprocessing
from xgboost import XGBClassifier
from tensorflow import keras
from keras.layers import Embedding, LSTM, Dense
from keras import models as mlp
from classes.data_exploring import ExploringData
from classes import tweet_cleaner as dataclean

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
    train = dataclean.tweet_cleaner(train, train_tweet,
                                    preprocessoptions=['noise', 'short_words', 'stop_words', 'rare_words',
                                                       'common_words', 'tokenization', 'lower_case'])
    # splitting data into training and validation set
    xtrain, xvalid, ytrain, yvalid = train_test_split(train[train_tweet], train[train_label], random_state=42,
                                                      test_size=0.3)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    ytrain = encoder.fit_transform(ytrain)
    yvalid = encoder.fit_transform(yvalid)

    # >>> COUNT VECTORIZER >>>
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train[train_tweet])
    xtrain_count = count_vect.transform(xtrain)
    xvalid_count = count_vect.transform(xvalid)
    # <<< COUNT VECTORIZER <<<

    # >>> TF-IDF WORD LEVEL >>>
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=2500)
    tfidf_vect.fit(train[train_tweet])
    xtrain_tfidf = tfidf_vect.transform(xtrain)
    xvalid_tfidf = tfidf_vect.transform(xvalid)
    # <<< TF-IDF WORD LEVEL <<<

    # >>> TF-IDF NGRAM-LEVEL >>>
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                       max_features=2500)
    tfidf_vect_ngram.fit(train[train_tweet])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(xtrain)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(xvalid)
    # <<< TF-IDF NGRAM-LEVEL <<<

    # >>> TF-IDF CHARACTER LEVEL >>>
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                             max_features=2500)
    tfidf_vect_ngram_chars.fit(train[train_tweet])
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(xtrain)
    xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(xvalid)
    # <<< TF-IDF CHARACTER LEVEL <<<

    # >>> BAG OF WORDS >>>
    train_bow = CountVectorizer(max_features=2500, lowercase=True, ngram_range=(1, 3), analyzer="word")
    train_bow.fit(train[train_tweet])
    xtrain_bow = train_bow.transform(xtrain)
    xvalid_bow = train_bow.transform(xvalid)
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

    def train_model(classifier, model_name, feature_name, feature_vector_train, label, feature_vector_valid,
                    neural_net=False):
        # fit the training dataset on the classifier
        if storemodel:
            model = classifier
            model.fit(feature_vector_train, label)

            # predict the labels on validation dataset
            predictions = model.predict(feature_vector_valid)
            filename=('data/results/stored_trained_models/'+ model_name +'_'+ feature_name+'.sav')
            joblib.dump(model, filename)
            entries.append((datasetname, model_name, metrics.accuracy_score(predictions, yvalid), feature_name))
            return metrics.accuracy_score(predictions, yvalid)
        else:
            model = classifier
            model.fit(feature_vector_train, label)

            # predict the labels on validation dataset
            predictions = model.predict(feature_vector_valid)
            accuracy=metrics.accuracy_score(predictions, yvalid)
            print(datasetname,model_name,accuracy)
            entries.append((datasetname, model_name,accuracy, feature_name))
            return metrics.accuracy_score(predictions, yvalid)

    def create_lstm():
        input_dim = xtrain_bow.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        history = model.fit(xtrain_bow, ytrain,
                            epochs=100,
                            verbose=False,
                            validation_data=(xvalid_bow, yvalid),
        batch_size = 10)
        return model

        loss, accuracy = model.evaluate(xtrain_bow, ytrain, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(xvalid_bow, yvalid, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))




    create_lstm()
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



    # >>> TRADITIONAL MACHINE LEARNING METHODS >>>

    #[train_model(model,model.__class__.__name__, feature.name, feature.xtrain, ytrain, feature.xvalid)for model in
    #models for feature in featureList]
    # <<< TRADITIONAL MACHINE LEARNING METHODS <<<


    # >>> DEEP LEARNING >>>

    # <<< DEEP LEARNING <<<




    # [train_model(classifier, classifier.clname, feature.name, feature.xtrain, ytrain, feature.xvalid) for classifier in
    #  classifierList for feature in featureList]
    #[train_model(classifier.classifiermodel,classifier.clname,"Word Embeddings",xtrain_bow.shape[1],ytrain,
    #             xvalid_bow.shape[1],is_neural_net=True)for classifier in classifierList]
    # <<< DEEP LEARNING <<<

    # >>> WRITE RESULTS >>>
    cv_df = pd.DataFrame(entries,columns=['dataset','model_name', 'accuracy','feature'])
    # print(cv_df)
    cv_df.to_csv('data/results/accuracy_table/all_results_from_training.csv',mode='a', index_label=False, header=False,
                 index=False)
    # <<< WRITE RESULTS <<<


    end = time.time()
    result = (end - start) / 60
    print("time to train "+datasetname+" took: ", str(result), " minutes")