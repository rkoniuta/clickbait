import sys

from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# https://www.kaggle.com/c/clickbait-news-detection/data
data = pd.read_csv("train.csv")

# replace all non alphanumeric, convert lower, split into list
data['title'] = data['title'].str.replace('\W', ' ')
data['title'] = data['title'].str.lower()
data["title"] = data['title'].str.split()

# calculate the set of all vocab
bad_rows = []
vocab = set()
vocab_seen = set()

for i, sms in tqdm(zip(data["id"], data['title'])):
    try:
        for word in sms:
            word = word.lower()
            if word in vocab_seen:
                vocab.add(word)
            else:
                vocab_seen.add(word)
    except Exception:
        bad_rows.append(i)

data = data.drop(bad_rows)
data, test = train_test_split(data, test_size=0.2)

vocab = list(vocab) # set of all words that appear twice or more
n_vocab = len(vocab)
freq_clickbait = {word:0 for word in vocab + ['unk']}
freq_news = {word:0 for word in vocab + ['unk']}
vocab = {word: i for i, word in enumerate(vocab)}

# Used NP for speed. Literally 100x faster than Pandas
rows = len(data['label'])
cols = n_vocab + 3 # +3 because need length, unk, and label parameters
np_data = np.zeros((rows, cols), dtype=int)

for i, (title, label) in tqdm(enumerate(zip(data['title'], data['label']))):
    # vectorize words to numbers
    np_data[i][0] = len(title)
    for word in title:
        if word in vocab:
            np_data[i][vocab[word]+1] += 1
        else:
            np_data[i][-2] += 1

    if label == 'news':
        np_data[i][-1] = 0
    elif label == 'clickbait':
        np_data[i][-1] = 1

clickbait = np_data[np_data[:,-1] == 1]
news = np_data[np_data[:,-1] == 0]

# P(clickbait) and P(news)
prob_clickbait = len(clickbait) / len(np_data)
prob_news = len(news) / len(np_data)

# N_clickbait
n_clickbait = sum(clickbait[:,0])

# N_news
n_news = sum(news[:,0])

# Laplace smoothing
alpha = 1/float("inf")

# probabilities for each word in vocab
for word in tqdm(vocab):
    word_clickbait = sum(clickbait[:,vocab[word]+1])
    word_clickbait_prob = (word_clickbait + alpha) / (n_clickbait + alpha*n_vocab)
    freq_clickbait[word] = word_clickbait_prob

    word_news = sum(news[:,vocab[word]+1])
    word_news_prob = (word_news + alpha) / (n_news + alpha*n_vocab)
    freq_news[word] = word_news_prob

# probability for unk
word_clickbait = sum(clickbait[:,-2])
word_clickbait_prob = (word_clickbait + alpha) / (n_clickbait + alpha*n_vocab)
freq_clickbait["unk"] = word_clickbait_prob

word_news = sum(news[:,-2])
word_news_prob = (word_news + alpha) / (n_news + alpha*n_vocab)
freq_news["unk"] = word_news_prob


def classify(title):
    prob_clickbait_instance = prob_clickbait
    prob_news_instance = prob_news

    for word in title:
        if word in freq_clickbait:
            prob_clickbait_instance *= freq_clickbait[word]
        else:
            prob_clickbait_instance *= freq_clickbait["unk"]
        if word in freq_news:
            prob_news_instance *= freq_news[word]
        else:
            prob_news_instance *= freq_news["unk"]

    if prob_news_instance >= prob_clickbait_instance:
        return "news"
    elif prob_news_instance < prob_clickbait_instance:
        return "clickbait"


X = data['title']
y = data['label']

predicted = []
for title in X:
    predicted.append(classify(title))

correct = 0
for pred, real in zip(predicted, y):
    if pred == real:
        correct += 1

print(f"Train examples predicted correctly: {correct/len(y)}")

data = test
X = data['title']
y = data['label']

predicted = []
for title in X:
    predicted.append(classify(title))

correct = 0
for pred, real in tqdm(zip(predicted, y)):
    if pred == real:
        correct += 1
print(f"Test examples predicted correctly: {correct/len(y)}")
