# Now I want to see how well the given sentiments are distributed across
# the train dataset. One way to accomplish this task is by understanding
# the common words by plotting wordclouds.
# A wordcloud is a visualization wherein the most frequent words appear
# in large size and the less frequent words appear in smaller sizes.
# Let’s visualize all the words our data using the wordcloud plot.
from wordcloud import WordCloud
    import matplotlib.pyplot as plt
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

def explore_impact_of_hashtags(train,train_tweet,train_label):
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
    HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

    # unnesting list
    HT_regular = sum(HT_regular, [])
    HT_negative = sum(HT_negative, [])

