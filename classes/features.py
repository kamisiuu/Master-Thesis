#
# AUTHOR: KAMIL LIPSKI
#
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing import text, sequence
import numpy as np
# START OF COUNT VECTORS AS FEATURES
    # creates a count vectorizer object and transform the training and validation data using count vectorizer object
class FeatureClass:
    def __new__(cls,train,train_tweet,xtrain,xvalid):
        cls.train=train
        cls.train_tweet=train_tweet
        cls.xtrain=xtrain
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
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3),
                                           max_features=2500)
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
        vector_dataset = open('data/pretrained_vectors/wiki-news-300d-1M.vec')
        [() for line in vector_dataset]
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

        return featureList
            #################################End of List of Features##############################################
