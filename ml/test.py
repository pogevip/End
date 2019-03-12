import jieba.analyse as analyse
import numpy as np

count = 0

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
corpus=["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]

vectorizer = CountVectorizer()
count = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

tfidf = TfidfTransformer()
res = tfidf.fit_transform(count)
print(res)


if __name__ == '__main__':
    pass