import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

labels = ["news", "clickbait"]
d = pd.read_csv("clickbait_data.csv")
d0 = [len(headline.split()) for headline, label in zip(d["headline"], d["clickbait"]) if label == 0]
d1 = [len(headline.split()) for headline, label in zip(d["headline"], d["clickbait"]) if label == 1]

plt.bar(labels,[len(d0), len(d1)])
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.title("Data breakdown by label (Dataset 2)")
plt.show()

d0 = {k: v for k, v in Counter(d0).items()}
d1 = {k: v for k, v in Counter(d1).items()}

plt.bar(d1.keys(), d1.values(), color='r')
plt.bar(d0.keys(), d0.values(), color='g')
plt.title("Histogram of title length (Dataset 2)")
plt.xlabel("red = clickbait, green = news")
plt.ylabel("frequency")

plt.show()

d = pd.read_csv("train.csv")
d0 = [len(headline.split()) for headline, label in zip(d["title"], d["label"]) if label == "news" and len(headline.split()) < 50]
d1 = [len(headline.split()) for headline, label in zip(d["title"], d["label"]) if label == "clickbait" and len(headline.split()) < 50]


plt.bar(labels,[len(d0), len(d1)])
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.title("Data breakdown by label (Dataset 1)")
plt.show()

d0 = {k: v for k, v in Counter(d0).items()}
d1 = {k: v for k, v in Counter(d1).items()}

plt.bar(d0.keys(), d0.values(), color='g')
plt.bar(d1.keys(), d1.values(), color='r')
plt.title("Histogram of title length (Dataset 1)")
plt.xlabel("red = clickbait, green = news")
plt.ylabel("frequency")

plt.show()
