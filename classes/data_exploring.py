# Now I want to see how well the given sentiments are distributed across
# the train dataset. One way to accomplish this task is by understanding
# the common words by plotting wordclouds.
# A wordcloud is a visualization wherein the most frequent words appear
# in large size and the less frequent words appear in smaller sizes.
# Let’s visualize all the words our data using the wordcloud plot.
from threading import Thread

import pandas as pd
import seaborn as sns
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

class ExploringData:
    # this function constructs a class ExploringData
    def __init__(self,data,data_text,data_label):
        self.data=data
        self.data_text=data_text
        self.data_label=data_label

    # This function counts number of categories within a column
    def count_categories_within_column(self,plot=False):
        print('\nNumber of each category within column\n',
              self.data[self.data_label].value_counts())
        if plot:
            plot_size = plt.rcParams["figure.figsize"]
            print(plot_size[0])
            print(plot_size[1])

            plot_size[0] = 8
            plot_size[1] = 6
            plt.rcParams["figure.figsize"] = plot_size
            self.data[self.data_label].\
                value_counts().plot(kind='pie', autopct='%1.0f%%',
                                    colors=["blue",  "red"])

    # This function counts number of missing values within whole dataset
    def checkfor_missingvalues(self):
        print ('\nMissing Values in dataset\n',self.data.isnull().sum())

    # This function shows a diagram with common words used in
    # the tweets: WordCloud
    def explore_common_words(self):
        all_words=' '.join([text for text in self.data[self.data_text]])

        wordcloud=WordCloud(width=800, height=500, random_state=21,
                        max_font_size=110).generate(all_words)

        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    # This function plots diagram with words in non racist/sexist tweets
    def explore_non_racist_sexist_tweets(self):
        normal_words = ' '.join([text for text in self.data[self.data_text]
        [self.data[self.data_label] == 0]])
        wordcloud = WordCloud(width=800, height=500, random_state=21,
                              max_font_size=110).generate(normal_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.title('Words in non racist/sexist tweets')
        plt.show()

    # This function plots diagram with racist/Sexist Tweets
    def explore_racist_sexist_tweets(self):
        negative_words = ' '.join([text for text in
                                   self.data[self.data_text]
        [self.data[self.data_label] == 1]])
        wordcloud = WordCloud(width=800, height=500,
                          random_state=21, max_font_size=110).\
            generate(negative_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    # Understanding the impact of Hashtags on tweets sentiment
    def explore_hashtags(self):
        def hashtag_extract(x):
            hashtags = []
            # Loop over the words in the tweet
            for i in x:
                ht = re.findall(r"#(\w+)", i)
                hashtags.append(ht)
            return hashtags
        try:
            # extracting hashtags from non racist/sexist tweets
            HT_regular = hashtag_extract(
                self.data[self.data_text][self.data[self.data_label]==0])
            # extracting hashtags from racist/sexist tweets
            HT_negative = hashtag_extract(
                self.data[self.data_text][self.data[self.data_label]==1])

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
            e = pd.DataFrame({'Hashtag': list(b.keys()),'Count':
                list(b.values())})
            # selecting top 10 most frequent hashtags
            e = e.nlargest(columns="Count", n=10)
            plt.figure(figsize=(16, 5))
            ax = sns.barplot(data=e, x="Hashtag", y="Count")
            ax.set(ylabel='Count')
            plt.title('Racist/Sexist Tweets of HASHTAGS')
            plt.show()
        except Exception:
            print ("there are no hashtags in your data")
            pass
        else:pass


    # this method runs all previous methods
    def runall(self):
        self.count_categories_within_column(plot=True)
        self.checkfor_missingvalues()
        self.explore_common_words()
        self.explore_non_racist_sexist_tweets()
        self.explore_racist_sexist_tweets()
        self.explore_hashtags()

