from sklearn.naive_bayes import GaussianNB, MultinomialNB
import sys
import faulthandler
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

'''
Store some helper functions up here
'''

def train_results(preds):
    return "Training Accuracy:", accuracy_score(y_train,preds)

def test_results(preds):
    return "Testing Accuracy:", accuracy_score(y_test,preds)

# Import the dataset
df = pd.read_csv("train.csv")

# Use a stopwords dataset to remove common words that could distort the dataset
stopwords_list = stopwords.words('english')

# Clean out the rows with missing stuff
bad_rows = []
for i, sms in tqdm(zip(df["id"], df['title'])):
    try:
        for word in sms:
            if word is None:
                bad_rows.append(i)
                break
    except Exception:
        bad_rows.append(i)
df = df.drop(bad_rows)
bad_rows = []

for i, sms in tqdm(zip(df["id"], df['text'])):
    try:
        for word in sms:
            if word is None:
                bad_rows.append(i)
                break
    except Exception:
        bad_rows.append(i)

# Drop the stuff we don't want
df = df.drop(bad_rows)
df['label_int'] = df['label'].map(lambda x: 2 if x == 'news' else 1 if x == 'clickbait' else 0)
df.drop(columns='label', inplace=True)


# Create processed datasets
features = df.drop(columns='label_int')
y = df['label_int']

# Show the counts
print(f'Display how many of each type there are: {y.value_counts()}')

X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=40)

# Vectorize the titles
tfidf = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1,2)) # ngram_range of (1,2) produces the best results in my testing
tfidf_title_train = tfidf.fit_transform(X_train['title'])
tfidf_title_test = tfidf.transform(X_test['title'])

X_train_ef = X_train.drop(columns=['title', 'text'])
X_test_ef = X_test.drop(columns=['title', 'text'])




X_train = sparse.hstack([X_train_ef, tfidf_title_train]).tocsr()
X_test = sparse.hstack([X_test_ef, tfidf_title_test]).tocsr()

clf = MultinomialNB(alpha=0.01) # This variable can be modifed and gets slightly diff results

clf.fit(X_train, y_train)

nb_train_preds = clf.predict(X_train)
nb_test_preds = clf.predict(X_test)

print(train_results(nb_train_preds))
print(test_results(nb_test_preds))