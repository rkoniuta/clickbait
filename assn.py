import sys

from tqdm import tqdm

import numpy as np
import pandas as pd


# TODO hold out a testing set
# TODO use full body text, not just title
# TODO use SKLEARN models
# TODO pickle this model for inference

# https://www.kdnuggets.com/2020/07/clickbait-filter-python-naive-bayes-scratch.html
data = pd.read_csv("train.csv")[:10000]

# replace all non alphanumeric, convert lower, split into list
data['title'] = data['title'].str.replace('\W', ' ')
data['title'] = data['title'].str.lower()
data["title"] = data['title'].str.split()

# calculate the set of all vocab
bad_rows = [] #TODO there are an excessive amount of "bad rows" aka poorly formatted or missing a title (~4000)
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

vocab = list(vocab) # set of all words that appear twice or more
n_vocab = len(vocab)
parameters_clickbait = {word:0 for word in vocab + ['unk']}
parameters_news = {word:0 for word in vocab + ['unk']}
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
print(len(clickbait))
print(len(news))

# P(clickbait) and P(news)
p_clickbait = len(clickbait) / len(np_data)
p_news = len(news) / len(np_data)

# N_clickbait
n_clickbait = sum(clickbait[:,0])

# N_news
n_news = sum(news[:,0])

# Laplace smoothing
alpha = 1/float("inf")

# probabilities for each word in vocab
for word in tqdm(vocab):
    n_word_given_clickbait = sum(clickbait[:,vocab[word]+1])
    p_word_given_clickbait = (n_word_given_clickbait + alpha) / (n_clickbait + alpha*n_vocab)
    parameters_clickbait[word] = p_word_given_clickbait

    n_word_given_news = sum(news[:,vocab[word]+1])
    p_word_given_news = (n_word_given_news + alpha) / (n_news + alpha*n_vocab)
    parameters_news[word] = p_word_given_news

# probability for unk
n_word_given_clickbait = sum(clickbait[:,-2])
p_word_given_clickbait = (n_word_given_clickbait + alpha) / (n_clickbait + alpha*n_vocab)
parameters_clickbait["unk"] = p_word_given_clickbait

n_word_given_news = sum(news[:,-2])
p_word_given_news = (n_word_given_news + alpha) / (n_news + alpha*n_vocab)
parameters_news["unk"] = p_word_given_news


def classify(title):
    p_clickbait_given_title = p_clickbait
    p_news_given_title = p_news

    for word in title:
        if word in parameters_clickbait:
            p_clickbait_given_title *= parameters_clickbait[word]
        else:
            p_clickbait_given_title *= parameters_clickbait["unk"]
        if word in parameters_news:
            p_news_given_title *= parameters_news[word]
        else:
            p_news_given_title *= parameters_news["unk"]

    if p_news_given_title >= p_clickbait_given_title:
        return "news"
    elif p_news_given_title < p_clickbait_given_title:
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

print(len(predicted) == len(y))
print(correct/len(y))