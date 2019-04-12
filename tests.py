# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
#
# from models.SVCgridSearch import paramTunSVC
# from classes.data_exploring import ExploringData
# from classes.data_upload import Upload
# train = pd.read_csv("data/dataset_1/train.csv", header='infer', index_col=None)
# #test = pd.read_csv("data/dataset_1/test.csv", header='infer', index_col=None)
#
# from classes.tweet_cleaner import tweet_cleaner
# #data = pd.read_csv('data/results/accuracy_table/all_results_from_training.csv')
#
#
# # count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# # xtrain_count =count_vect.fit_transform(train['SentimentText'])
# #
# # paramTunSVC(xtrain_count,train['Sentiment'])
#
#
#
# data = tweet_cleaner(train,'tweet')
# print(data)
# START TRAINING WITH DEEP NEURAL NETWORKS
classifierList = {'CNN': 'create_cnn()', 'RCNN': 'create_rcnn()', 'RNN-LSTM': 'create_rnn_lstm()',
                  'RNN-GRU': 'create_rnn_gru()', 'BIDIRECTIONAL-RNN': 'create_bidirectional_rnn()'}

for choicemodel in classifierList:
    #print (choicemodel)
    print( classifierList[choicemodel])
# from classes.grid_search_utility import grid_search_svm
# import gensim
# from gensim.models import Word2Vec
#
# #loading the downloaded model
# model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
#
# #the model is loaded. It can be used to perform all of the tasks mentioned above.
#
# # getting word vectors of a word
# dog = model['dog']
#
# #performing king queen magic
# print(model.most_similar(positive=['woman', 'king'], negative=['man']))
#
# #picking odd one out
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))
#
# #printing similarity index
# print(model.similarity('woman', 'man'))


# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', MultinomialNB())])
#
# tuned_parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
#     'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     'clf__alpha': [1, 1e-1, 1e-2]
# }
#
#
# x_train, x_test, y_train, y_test = train_test_split(train['tweet'], train['label'], random_state=42,
#                                                test_size=0.3)
#
#
# from sklearn.metrics import classification_report
# clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
# clf.fit(x_train, y_train)
#
# print(classification_report(y_test, clf.predict(x_test), digits=4))
#
#
#
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC
#
# print(__doc__)
#
# # Loading the Digits dataset
# digits = datasets.load_digits()
#
# # To apply an classifier on this data, we need to flatten the image, to
# # turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# X = digits.images.reshape((n_samples, -1))
# y = digits.target
#
# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=0)
#
# # Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.