# data taken from images/ directory, after running sentiment.py
import matplotlib.pyplot as plt
import numpy as np

labels          = ["sadness", "fear", "disgust", "joy", "surprise", "anger"]
freq_news       = [.12324, .05843, .01474, .01881, .000499, .00006249]
freq_clickbait  = [.09538, .011063, .055194, .003687, .01875, .000375]
  
X_axis = np.arange(len(labels))
  
plt.bar(X_axis - 0.2, list(map(lambda x: x/sum(freq_news), freq_news)), 0.4, label = 'news')
plt.bar(X_axis + 0.2, list(map(lambda x: x/sum(freq_clickbait), freq_clickbait)), 0.4, label = 'clickbait')
  
plt.xticks(X_axis, labels)
plt.xlabel("Emotion")
plt.ylabel("Relative frequency")
plt.title("Relative requency of detected emotions")
plt.legend()
plt.show()


# tone analysis
labels          = ["neutral", "positive", "negative"]
freq_news       = [.61789, .06899, .313105]
freq_clickbait  = [.56219, .26872, .16915]

X_axis = np.arange(len(labels))
  
plt.bar(X_axis - 0.2, list(map(lambda x: x/sum(freq_news), freq_news)), 0.4, label = 'news')
plt.bar(X_axis + 0.2, list(map(lambda x: x/sum(freq_clickbait), freq_clickbait)), 0.4, label = 'clickbait')
  
plt.xticks(X_axis, labels)
plt.xlabel("Tone")
plt.ylabel("Relative frequency")
plt.title("Relative requency of tone")
plt.legend()
plt.show()
