import numpy

from envir import upload_data as upld

import pandas as pd
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import layers, models, optimizers
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

train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)

# splitting data into training and validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(train['tweet'], train['label'], random_state=42, test_size=0.3)
#train = pr.data_clean(train)

# create model without any feature


# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train['tweet'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(xtrain)
xvalid_count =  count_vect.transform(xvalid)

# # Bag og words feature
train_bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow.fit(train['tweet'])

xtrain_bow = train_bow.transform(xtrain)
xvalid_bow = train_bow.transform(xvalid)

# tf-idf feature
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train['tweet'])
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

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier


SNN = create_model_architecture(xtrain_tfidf.shape[1])



models=[MultinomialNB(),SVC(kernel='linear'),LogisticRegression( solver='lbfgs'),KNeighborsClassifier(),RandomForestClassifier(),XGBClassifier(),
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


snn_accuracy = train_model(SNN,xtrain_tfidf,ytrain,xvalid_tfidf, is_neural_net=True)[0]


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
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model

# load the pre-trained word-embedding vectors
embeddings_index = {}
for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(train['tweet'])
word_index = token.word_index
# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(xtrain), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(xvalid), maxlen=70)
# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, ytrain, valid_seq_x, is_neural_net=True)[0]
print
"CNN, Word Embeddings", accuracy

#
# CV = 10
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, xtrain_count, ytrain, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
# print(cv_df.groupby('model_name').accuracy.mean())