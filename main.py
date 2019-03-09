import numpy
import pandas as pd
import warnings

from classes import data_cleaning as preproc
from classes import data_upload as upld
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import layers, models as mdl, optimizers
from keras.preprocessing import sequence, text
from sklearn import metrics, model_selection, preprocessing
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# resources:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://cmdlinetips.com/2018/11/string-manipulations-in-pandas/
# w3schools.com/python


#Origin import, tranforming into csv
#train = pd.read_csv("../data/training.txt", sep="\t", header=None)
#test = pd.read_csv("../data/testdata.txt", delim_whitespace=False, delimiter="\n", header=None)
#train.columns = ["label", "tweet"]
#test.columns = ["tweet"]
#df_train = DataFrame(train, columns=train.columns)
#df_test = DataFrame(test,columns=test.columns)
#df_train.to_csv('data/train.csv',index=None,header=True)
#df_test.to_csv('data/test.csv',index=None,header=True)


#

#train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
# test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)

train = pd.read_csv("data/dataset_2/train.csv", header='infer',index_col=None)
#test = pd.read_csv("data/dataset_2/test.csv", delimiter=None, header='infer', names=None, index_col=None,  )

#preprocessing
#train = preproc.data_clean(train,"tweet)
print("not preprocessed",train, "\n")
train = preproc.clean_data(train,"SentimentText")

def startTraining(train,train_tweet,train_label, gridsearch=False, multitrain=False):
    # splitting data into training and validation set
    xtrain, xvalid, ytrain, yvalid = train_test_split(train[train_tweet], train[train_label], random_state=42, test_size=0.3)
    # train = pr.data_clean(train)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(train[train_tweet])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(xtrain)
    xvalid_count =  count_vect.transform(xvalid)

    # # Bag og words feature
    train_bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
    train_bow.fit(train[train_tweet])

    xtrain_bow = train_bow.transform(xtrain)
    xvalid_bow = train_bow.transform(xvalid)

    # tf-idf feature
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(train[train_tweet])
    xtrain_tfidf =  tfidf_vect.transform(xtrain)
    xvalid_tfidf =  tfidf_vect.transform(xvalid)



    # Lets implement these models and understand their details.
    # The following function is a utility function which can be used to train a model.
    # It accepts the classifier, feature_vector of training data, labels of training data and
    # feature vectors of valid data as inputs. Using these inputs, the model is trained and accuracy score is computed.

    def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        train_model_array=[metrics.accuracy_score(predictions, yvalid),predictions]
        return train_model_array


    #SHALLOW NEURAL NETWORKS
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

    if multitrain:
        models = [MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), SVC(kernel='linear'),
                  LogisticRegression(solver='lbfgs'), KNeighborsClassifier(), RandomForestClassifier(), XGBClassifier(),
                  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1)]
        entries=[]
        for model in models:
            # nofeatures_accuracy = train_model(model,xtrain,ytrain,xvalid)[0]

            # count vectorizer object
            cvo_accuracy = train_model(model,xtrain_count,ytrain,xvalid_count)[0]

            # bag of words
            bow_accuracy = train_model(model, xtrain_bow, ytrain, xvalid_bow)[0]
            model_name= model.__class__.__name__

            # TF-IDF
            tfidf_accuracy = train_model(model, xtrain_tfidf, ytrain, xvalid_tfidf)[0]

            entries.append((model_name,  cvo_accuracy, bow_accuracy,tfidf_accuracy))

        cv_df = pd.DataFrame(entries, columns=['model_name','cvo','bog', 'tfidf_word_lvl'])
        print(cv_df)
    else:
        CV=10
        nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        accuracies = cross_val_score(nb, xtrain_count, ytrain, scoring='accuracy', cv=CV)
        print(accuracies)
    #SNN = create_model_architecture(xtrain_tfidf.shape[1])
    #models=[MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),SVC(kernel='linear'),LogisticRegression( solver='lbfgs'),KNeighborsClassifier(),RandomForestClassifier(),XGBClassifier(),MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1)]
    # entries=[]
    # for model in models:
    #     # nofeatures_accuracy = train_model(model,xtrain,ytrain,xvalid)[0]
    #
    #     # count vectorizer object
    #     cvo_accuracy = train_model(model,xtrain_count,ytrain,xvalid_count)[0]
    #
    #     # bag of words
    #     bow_accuracy = train_model(model, xtrain_bow, ytrain, xvalid_bow)[0]
    #     model_name= model.__class__.__name__
    #
    #     # TF-IDF
    #     tfidf_accuracy = train_model(model, xtrain_tfidf, ytrain, xvalid_tfidf)[0]
    #
    #     entries.append((model_name,  cvo_accuracy, bow_accuracy,tfidf_accuracy))
    #
    # cv_df = pd.DataFrame(entries, columns=['model_name','cvo','bog', 'tfidf_word_lvl'])
    # print(cv_df)
    #
    #
    # snn_accuracy = train_model(SNN, xtrain_tfidf, ytrain, xvalid_tfidf, is_neural_net=True)[0]
    # print(snn_accuracy)

    nb= MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True).fit(xtrain_count,ytrain)
    prediction = nb.predict(xvalid_count)
    print(f1_score(yvalid, prediction))
    entries=[]
    entries.append()
#     CV = 2
#     cv_df = pd.DataFrame(index=range(CV * len(models)))
#     entries = []
#     for model in models:
#         model_name = model.__class__.__name__
#         accuracies = cross_val_score(model, xtrain_count, ytrain, scoring='accuracy', cv=CV)
#         for fold_idx, accuracy in enumerate(accuracies):
#             entries.append((model_name, fold_idx, accuracy))
#     cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
#     print(cv_df.groupby('model_name').accuracy.mean())
#
#
# startTraining(train,"tweet","label")

startTraining(train, "SentimentText", "Sentiment")


