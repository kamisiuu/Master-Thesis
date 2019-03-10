# Now I want to see how well the given sentiments are distributed across
# the train dataset. One way to accomplish this task is by understanding
# the common words by plotting wordclouds.
# A wordcloud is a visualization wherein the most frequent words appear
# in large size and the less frequent words appear in smaller sizes.
# Let’s visualize all the words our data using the wordcloud plot.
import pandas as pd
import seaborn as sns
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
#  Understanding the common words used in the tweets: WordCloud
def explore_common_words(train,train_tweet):
    all_words = ' '.join([text for text in train[train_tweet]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

# Words in non racist/sexist tweets
def explore_non_racist_sexist_tweets(train,train_tweet,train_label):
    normal_words = ' '.join([text for text in train[train_tweet][train[train_label] == 0]])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Words in non racist/sexist tweets')
    plt.show()


# Racist/Sexist Tweets
def explore_racist_sexist_tweets(train,train_tweet,train_label):
    negative_words = ' '.join([text for text in train[train_tweet][train[train_label] == 1]])
    wordcloud = WordCloud(width=800, height=500,
                          random_state=21, max_font_size=110).generate(negative_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

# Understanding the impact of Hashtags on tweets sentiment

def explore_hashtags(train,train_tweet,train_label):
    def hashtag_extract(x):
        hashtags = []
        # Loop over the words in the tweet
        for i in x:
            ht = re.findall(r"#(\w+)", i)
            hashtags.append(ht)

        return hashtags

    # extracting hashtags from non racist/sexist tweets

    HT_regular = hashtag_extract(train[train_tweet][train[train_label] == 0])

    # extracting hashtags from racist/sexist tweets
    HT_negative = hashtag_extract(train[train_tweet][train[train_label] == 1])

    # unnesting list
    HT_regular = sum(HT_regular, [])
    HT_negative = sum(HT_negative, [])
    a = nltk.FreqDist(HT_regular)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 10 most frequent hashtags
    d = d.nlargest(columns="Count", n=10)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=d, x="Hashtag", y="Count")
    ax.set(ylabel='Count')
    plt.title('Non-Racist/Sexist Tweets of HASHTAGS')
    plt.show()

    b = nltk.FreqDist(HT_negative)
    e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
    # selecting top 10 most frequent hashtags
    e = e.nlargest(columns="Count", n=10)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=e, x="Hashtag", y="Count")
    ax.set(ylabel='Count')
    plt.title('Racist/Sexist Tweets of HASHTAGS')
    plt.show()

