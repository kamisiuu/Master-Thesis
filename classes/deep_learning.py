
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder



from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils

#https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
#https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-10-neural-network-with-a6441269aa3c
# https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67

train_1 = pd.read_csv("../data/dataset_1/train.csv", header='infer', index_col=None)
test= pd.read_csv("../data/dataset_1/test.csv", header='infer', index_col=None)
df = pd.read_csv("../data/dataset_2/train.csv", header='infer',index_col=None)
test_2 = pd.read_csv("../data/dataset_2/test.csv", delimiter=None, header='infer', names=None, index_col=None, encoding='latin-1')


train_size = int(len(df) * .7)
train_posts = df['SentimentText'][:train_size]
train_tags = df['Sentiment'][:train_size]

test_posts = df['SentimentText'][train_size:]
test_tags = df['Sentiment'][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts)  # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 2

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])