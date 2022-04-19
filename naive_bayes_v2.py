from sklearn.naive_bayes import GaussianNB, MultinomialNB
import sys
import faulthandler
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score

import numpy as np
import pandas as pd
#from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

'''
Store some helper functions up here
'''
#creating a function to call after each model iteration to print accuracy and recall scores for test and train
def train_results(preds):
    return "Training Accuracy:", accuracy_score(y_train,preds)

def test_results(preds):
    return "Testing Accuracy:", accuracy_score(y_test,preds)
    
faulthandler.enable()
# Import the dataset
df = pd.read_csv("train.csv")[:5000]

# Use a stopwords dataset to remove common words that could distort the dataset
stopwords_list = stopwords.words('english')

# Clean out the rows with missing stuff and stuff categorized as 'other'
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
# Drop the stuff we don't want to use
df = df.drop(bad_rows)
# Could experiment with leaving other tag in.
#df.drop(df[df['label'] == 'other'].index, inplace = True)
df['label_int'] = df['label'].map(lambda x: 2 if x == 'news' else 1 if x == 'clickbait' else 0)
# drop the text version of label and the article text
df.drop(columns='label', inplace=True)
#df.drop(columns='text', inplace=True)

# Create the processed datasets
features = df.drop(columns='label_int')
y = df['label_int']

#show the counts
print(f'Display how many of each type there are: {y.value_counts()}')

X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=20)

# Vectorize the titles
tfidf = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1,2))
tfidf_title_train = tfidf.fit_transform(X_train['title'])
tfidf_text_train = tfidf.fit_transform(X_train['text'])
tfidf_title_test = tfidf.transform(X_test['title'])
tfidf_text_test = tfidf.transform(X_test['text'])

X_train_ef = X_train.drop(columns=['title', 'text'])
X_test_ef = X_test.drop(columns=['title', 'text'])

print(X_train_ef)

from scipy import sparse

X_train = sparse.hstack([X_train_ef, tfidf_title_train, tfidf_text_train]).tocsr()
X_test = sparse.hstack([X_test_ef, tfidf_title_test, tfidf_text_test]).tocsr()

#print(X_train_ef)
#print(tfidf_title_train)
#print(X_train)
#X_train = [X_train_ef, tfidf_title_train]
#X_test = [X_test_ef, tfidf_title_test]
print('passed conversion')
clf = GaussianNB()
print('about to fit')

clf.fit(X_train.toarray(), y_train)
print('model fitted')

nb_train_preds = clf.predict(X_train.toarray())
nb_test_preds = clf.predict(X_test.toarray())

print(train_results(nb_train_preds))
print(test_results(nb_test_preds))