import random

from collections import defaultdict
from csv import reader

import torch

from pysentimiento import create_analyzer
from tqdm import tqdm
from transformers import pipeline


analyzer = create_analyzer(task="sentiment", lang="en") # replace sentiment with emotion to reproduce results.

data = []
# download from kaggle
with open("clickbait_data.csv") as f:
    csv_reader = reader(f)
    for row in csv_reader:
        data.append(row)

data = [[row[0], int(row[1])] for row in data[1:]]

clickbait = [row for row in data[1:] if row[1] == 1]
n_clickbait = len(clickbait)
d_clickbait = defaultdict(lambda: 0)

news = [row for row in data[1:] if row[1] == 0]
n_news = len(news)
d_news = defaultdict(lambda: 0)

for sentence, label in tqdm(data):
    sentiment = analyzer.predict(sentence)
    if label == 0:
        d_news[sentiment.output] += 1
    elif label == 1:
        d_clickbait[sentiment.output] += 1

for key, count in d_news.items():
    print(f"class: news, label: {key}, percent: {count/n_news}")

for key, count in d_clickbait.items():
    print(f"class: clickbait, label: {key}, percent: {count/n_clickbait}")
        
