import numpy as np
import pandas as pd
import gensim
import multiprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


rev = []
nfeat = 200
df = pd.read_csv("data/train.csv", header = -1)
for r in df[0].values.tolist():
        rev.append(r.split())

w2v = gensim.models.word2vec.Word2Vec(sg = 1, min_count = 2, window =6, size = nfeat, workers = multiprocessing.cpu_count())
w2v.build_vocab(sentences = rev)
w2v.train(rev, total_examples = len(rev), epochs= 5)

def rev2vec(reviews, model, nf):
    revFeat = np.zeros((len(reviews),nf))
    for r in range(len(reviews)):
        rv = np.zeros(nf)
        n = 0
        for word in reviews[r]:
            if (word in indexing):
                rv = rv + model[word]
                n += 1
        revFeat[r] = rv/n
    return revFeat

indexing = w2v.wv.index2word
revFeat = rev2vec(rev, w2v, nfeat)

ratings = df[1].values.tolist()

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(revFeat, ratings)

test = "data/test.csv"
df_test = pd.read_csv(test, header = -1)
test_sentences = []
for review in df_test[0].values.tolist():
    review = review.split()
    test_sentences.append(review)
Y = df_test[1].values.tolist()
testDataVecs = rev2vec(test_sentences, w2v, nfeat)

result = forest.predict(testDataVecs)
Y = df_test[1].values.tolist()


print(accuracy_score(Y, result))





