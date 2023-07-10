from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import wordcloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacy import displacy
import re
import os
import seaborn as sns
import tweepy



import pandas as pd
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter



# api_key="Qhd7FjJCgLfAEYM7AEB7DjDz0"
# api_secret="5tP9cEO5hMdKjHdtneFxW0yiuuCnrV4YvfkmVwzy0Zk0qRJIe9"
# access_token ="1333149297102110720-pVSdSuXdA1aErQ4u1jkcGLEOdhzOe0"
# access_secret="Pa4PkaoIZBgsRMTkQyIULANREqVaaACKir5h5rLCVuUwh"

access_token = "3251395693-QHNOSAdrkDHRBgG5TI8US5EPk434VKZ9eV6KJNa"
access_secret = "wcE4bQoGuNbzzdaTY3Uad25gm8h8QgtS7iIsyX0FJa4Tf"
api_key = "i0QOhnjQFaw1LEwwUxY7PeWrB"
api_secret = "IXZ9pzJK6ht579lX5dXlLDoBgWqRs7vrayzLKbeMQ8fxKDvqYv"

# @elonmusk


# build an api object to fetch tweets from twitter

# consumer_key = "************" #Your API/Consumer key
# consumer_secret = "*********" #Your API/Consumer Secret Key
# access_token = "***********"    #Your Access token key
# access_token_secret = "*************" #Your Access token Secret key

# Pass in our twitter API authentication key
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

tweets = api.search_tweets(q="Python",count=2)

l=[]
for tw in tweets:
    l.append(tw.text)

# print(type(tweets))
print(l)





# q="python"
# count=10
# tweets= api.search_tweets("python",10)
# for tw in tweets:
#     print(tw.text)
# scrapper = sntwitter.TwitterTweetScraper("#happy")
# for tweet in scrapper.get_items():
#     break;
#
# tweet=tweet.content
# # tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
# print(tweet)


# tweet='i am happy'
# precprcess tweet
tweet_words = []

for word in l:
    if word.startswith('@') and len(word) > 1:
        word = '@user'

    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l, s)
