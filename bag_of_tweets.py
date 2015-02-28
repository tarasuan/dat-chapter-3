import csv
import sys
import nltk
from nltk import FreqDist

# create rows from csv
f = open('twittersample.csv', 'rb')
rows = []
for row in csv.reader(f):
  rows.append(row)

# make a new list of sentiment text split into lists of words
tweets = []
for (sentiment, sentimenttext) in rows:
  sentimenttext_filtered = [e.lower() for e in sentimenttext.split() if len(e) >= 3]
  tweets.append((sentiment, sentimenttext_filtered))

f.close()

# make a list of just the words in the sentiment text
all_words = []

for (sentiment, sentimenttext_filtered) in tweets:
  all_words.extend(sentimenttext_filtered)

# make a frequency distribution list of all_words and a list of just the keys to be used to make the feature extractor
wordlist = nltk.FreqDist(all_words)
word_features = wordlist.keys()

# make the feature extractor
# not finished
document_words = set(tweets)
features = {}
for word in word_features:
  features['contains(%s)' % word] = (word in document_words)