#
# AUTHOR: KAMIL LIPSKI
#
import gensim
import pandas as pd
import numpy as np
import warnings
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import layers, models as mdl, optimizers
from sklearn import metrics, preprocessing
from classes.data_exploring import ExploringData
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from classes import tweet_cleaner as dataclean
from sklearn.externals import joblib
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import time
# resources:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://cmdlinetips.com/2018/11/string-manipulations-in-pandas/
# w3schools.com/python
# https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

train_1 = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
test_1 = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)
train_2 = pd.read_csv("data/dataset_2/train.csv", header='infer',index_col=None)
test_2 = pd.read_csv("data/dataset_2/test.csv", delimiter=None, header='infer', names=None, index_col=None, encoding='latin-1')


def Train(train,datasetname,train_tweet,train_label, clean=False, dataexplore=False, storemodel=False):
    # splitting data into training and validation set
    xtrain, xvalid, ytrain, yvalid = train_test_split(train[train_tweet], train[train_label], random_state=42,
                                                      test_size=0.3)
    if dataexplore:
        exp1= ExploringData(train,train_tweet,train_label)
        exp1.runall()

    #############################################PREPROCESSING START############################################
    train=dataclean.tweet_cleaner(train,train_tweet,preprocessoptions=[
        'noise','short_words','stop_words','rare_words','common_words','lemmatization','lower_case'])



    # START OF COUNT VECTORS AS FEATURES
    # creates a count vectorizer object and transform the training and validation data using count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train[train_tweet])
    xtrain_count = count_vect.transform(xtrain)
    xvalid_count = count_vect.transform(xvalid)
    # END OF COUNT VECTORS AS FEATURES

    # START OF TF-IDF VECTORS AS FEATURES
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=2500)
    tfidf_vect.fit(train[train_tweet])
    xtrain_tfidf = tfidf_vect.transform(xtrain)
    xvalid_tfidf = tfidf_vect.transform(xvalid)
    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), max_features=2500)
    tfidf_vect_ngram.fit(train[train_tweet])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(xtrain)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(xvalid)
    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                             max_features=2500)
    tfidf_vect_ngram_chars.fit(train[train_tweet])
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(xtrain)
    xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(xvalid)
    # END OF TF-IDF VECTORS AS FEATURES

    # START OF BAG OF WORDS FEATURE
    train_bow = CountVectorizer(max_features=2500, lowercase=True, ngram_range=(1, 3), analyzer="word")
    train_bow.fit(train[train_tweet])
    xtrain_bow = train_bow.transform(xtrain)
    xvalid_bow = train_bow.transform(xvalid)
    # END OF BAG OF WORDS FEATURE

    # START OF PRETRAINED WORDEMBEDDING FEATURE
    # load the pre-trained word-embedding vectors
    embeddings_index = {}

    for i, line in enumerate(open('data/pretrained_vectors/wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    # create a tokenizer
    token = text.Tokenizer()
    token.fit_on_texts(train[train_tweet])
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(xtrain), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(xvalid), maxlen=70)

    # create token-embedding mapping
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # END OF PRETRAINED WORDEMBEDDING FEATURE
    #############################################PREPROCESSING END############################################





    def train_model(classifier, model_name, feature_name, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier

        if storemodel:
            model = classifier
            model.fit(feature_vector_train, label)

            # predict the labels on validation dataset
            predictions = model.predict(feature_vector_valid)

            if is_neural_net:
                predictions = predictions.argmax(axis=-1)
            filename=('data/results/stored_trained_models/'+ model_name +'_'+ feature_name+'.sav')
            joblib.dump(model, filename)
            entries.append((datasetname, model_name, metrics.accuracy_score(predictions, yvalid), feature_name))
            return metrics.accuracy_score(predictions, yvalid)
        else:
            model = classifier
            model.fit(feature_vector_train, label)

            # predict the labels on validation dataset
            predictions = model.predict(feature_vector_valid)

            if is_neural_net:
                predictions = predictions.argmax(axis=-1)
            entries.append((datasetname, model_name, metrics.accuracy_score(predictions, yvalid), feature_name))
            return metrics.accuracy_score(predictions, yvalid)



    def create_model_architecture(input_size):
        # create input layer
        input_layer = layers.Input((input_size,), sparse=True)

        # create hidden layer
        hidden_layer = layers.Dense(100, activation="relu")(input_layer)

        # create output layer
        output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

        classifier = mdl.Model(inputs=input_layer, outputs=output_layer)
        classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        return classifier

    def create_cnn():
        # Add an Input Layer
        input_layer = layers.Input((70,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = mdl.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def create_rnn_lstm():
        # Add an Input Layer
        input_layer = layers.Input((70,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the LSTM Layer
        lstm_layer = layers.LSTM(100)(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = mdl.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def create_rnn_gru():
        # Add an Input Layer
        input_layer = layers.Input((70,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the GRU Layer
        lstm_layer = layers.GRU(100)(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = mdl.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def create_bidirectional_rnn():
        # Add an Input Layer
        input_layer = layers.Input((70,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the LSTM Layer
        lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = mdl.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def create_rcnn():
        # Add an Input Layer
        input_layer = layers.Input((70,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the recurrent layer
        rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)

        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = mdl.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        return model

    #################################Start of List of Features############################################
    # This class helps further in
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
    featureList.append(Feature('TF-IDF-NGRAM-CHARS', xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars))
    featureList.append(Feature('Word Embeddings Wiki', train_seq_x, valid_seq_x))

    #################################End of List of Features##############################################



    ####################START TRAINING WITH TRADITIONAL MACHINE LEARNING METHODS##########################
    models = [MultinomialNB(),LogisticRegression(),SGDClassifier(),KNeighborsClassifier(),RandomForestClassifier(),XGBClassifier()]
    entries = []
    [train_model(model,model.__class__.__name__, feature.name, feature.xtrain, ytrain, feature.xvalid)for model in models for feature in featureList]
    #####################END TRAINING WITH TRADITIONAL MACHINE LEARNING METHODS###########################

    #####################START OF TRAINING WITH SHALLOW NEURAL NETWORKS####################
    train_model(create_model_architecture(featureList[1].xtrain.shape[1]),"shallow neural network", featureList[1].name, featureList[1].xtrain, ytrain, featureList[1].xvalid,is_neural_net=True)
    ####################END OF TRAINING WITH SHALLOW NEURAL NETWORKS####################

    #################################START OF DEEP LEARNING#####################################################
    #################################Start of List of Deep Learning Classifiers#################################
    class Classifier:
        def __init__(self,clname, classifiermodel):
            self.clname=clname
            self.classifiermodel=classifiermodel

    classifierList=[]
    classifierList.append(Classifier('CNN',create_cnn()))
    classifierList.append(Classifier('RNN-LSTM',create_rnn_lstm()))
    classifierList.append(Classifier('RNN-GRU',create_rnn_gru()))
    classifierList.append(Classifier('RNN-Bidirectional',create_bidirectional_rnn()))
    classifierList.append(Classifier('RCNN',create_rcnn()))
    #################################End of List of Deep Learning Classifiers####################################

    ##########################Start of training with DEEP LEarning Models########################################
    [train_model(classifier.classifiermodel,classifier.clname,"Word Embeddings",train_seq_x,ytrain,valid_seq_x,is_neural_net=True)for classifier in classifierList]
    ##########################Start of training with DEEP LEarning Models########################################
    ###################################END OF DEEP LEARNING######################################################


    cv_df = pd.DataFrame(entries,columns=['dataset','model_name', 'accuracy','feature'])
    # print(cv_df)
    cv_df.to_csv('data/results/accuracy_table/all_results_from_training.csv',mode='a', index_label=False, header=False, index=False)













start = time.time()
#Train(train_1,"dataset_1","tweet","label")
Train(train_2,'dataset_2',"SentimentText","Sentiment",storemodel=True)
end = time.time()
result=(end-start)/60
print("time to train dataset_1 took: ",str(result)," minutes")
exit(0)