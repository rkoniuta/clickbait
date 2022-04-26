from sklearn.naive_bayes import GaussianNB, MultinomialNB
import sys

from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

'''
Store some helper functions up here
'''
#create a function to call after each model iteration to print scores
def train_results(preds):
    return "Training Accuracy:", accuracy_score(y_train,preds)

def test_results(preds):
    return "Testing Accuracy:", accuracy_score(y_test,preds)

# Import the dataset
df = pd.read_csv("with_engineeredfeat_data.csv")

# Use a stopwords dataset to remove common words that could distort the dataset
stopwords = stopwords.words('english')

# Create the processed datasets
features = df.drop(columns='class')
y = df['class']

#show the counts
print(f'Display how many of each type there are: {y.value_counts()}')

X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=40)

# Vectorize the titles
tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,2))
tfidf_title_train = tfidf.fit_transform(X_train['text'])
tfidf_title_test = tfidf.transform(X_test['text'])

X_train_ef = X_train.drop(columns='text')
X_test_ef = X_test.drop(columns='text')



X_train = sparse.hstack([X_train_ef, tfidf_title_train]).tocsr()
X_test = sparse.hstack([X_test_ef, tfidf_title_test]).tocsr()

clf = MultinomialNB(alpha=0.05)
clf.fit(X_train, y_train)

nb_train_preds = clf.predict(X_train)
nb_test_preds = clf.predict(X_test)

print(train_results(nb_train_preds))
print(test_results(nb_test_preds))