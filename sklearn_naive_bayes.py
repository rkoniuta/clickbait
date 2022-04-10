from sklearn.naive_bayes import GaussianNB
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd


# TODO: figure out what to do if the testing set uses a larger vocabulary than the training set
# Could just arbitrarily pick a large value but that seems unsatisfying

# https://www.kaggle.com/c/clickbait-news-detection/data
train_data = pd.read_csv("train.csv")[:10000]
test_data = pd.read_csv("train.csv")[10000:20000]
# replace all non alphanumeric, convert lower, split into list
train_data['title'] = train_data['title'].str.replace('\W', ' ')
train_data['title'] = train_data['title'].str.lower()
train_data["title"] = train_data['title'].str.split()

test_data['title'] = test_data['title'].str.replace('\W', ' ')
test_data['title'] = test_data['title'].str.lower()
test_data["title"] = test_data['title'].str.split()

def process(data, n_vocab=None):
    '''
    vectorize the titles to create an X value. Clean out the bad rows as well
    '''
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
    print(f'len of bad rows is {len(bad_rows)}')
    vocab = list(vocab) # set of all words that appear twice or more
    n_vocab = len(vocab) if n_vocab is None else n_vocab
    vocab = {word: i for i, word in enumerate(vocab)}

    # Used NP for speed. Literally 100x faster than Pandas
    rows = len(data['label'])
    cols = n_vocab + 2 # +3 because need length, unk params
    np_data = np.zeros((rows, cols), dtype=int)

    for i, (title, label) in tqdm(enumerate(zip(data['title'], data['label']))):
        # vectorize words to numbers
        np_data[i][0] = len(title)
        for word in title:
            if word in vocab:
                np_data[i][vocab[word]+1] += 1
            else:
                np_data[i][-2] += 1
    return np_data, data['label'], n_vocab

train_X, train_y, n_vocab = process(train_data) # np_data

clf = GaussianNB()
clf.fit(train_X, train_y)

test_X, test_y, _ = process(test_data, n_vocab)

print(clf.score(test_X, test_y))