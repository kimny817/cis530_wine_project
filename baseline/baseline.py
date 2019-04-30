'''
File name: baseline.py
Description:
Command to run it from terminal: python simple-baseline.py --train wine_train.csv --test wine_test.csv
'''

import argparse
import csv
from nltk import *
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, required=True)
parser.add_argument('--test', type=str, required=True)


class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# In[3]:


def classify(rating):
	# Assuming that 80 <= rating <=100
	return min(5, int((rating - 80) / 4) + 1)


def read_data(file_path, is_train):
	'''
		:param file_path: path to a file
				For example,
				-.data/train.csv
				-.data/test.csv
		:return: a tuple consisting of
				1. a feature vector that is a vectorized list of descriptions
				2. a list of labels (1,2,3,4,5)
	'''
	# stopwords_set = set(stopwords.words('english'))

	description_list = []
	labels = []

	for file in file_paths:
		with open(file, 'r') as f:
			csv_reader = csv.reader(f, delimiter=',')
			next(csv_reader)  # This skips the first row of the CSV file.

			for row in csv_reader:
				description = row[0]
				rating = int(row[1])
				# tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', description).split()
				# tokens = [t.lower() for t in tokens]
				# tokens = [lemmatizer(w) for w in tokens if w not in stopwords_set]
				labels.append(rating)
				description_list.append(description)

	return description_list, np.array(labels)


# In[5]:


def vectorize(list, is_ngram, n):
	vectorizer = CountVectorizer(ngram_range=(1, n)) if is_ngram else TfidfVectorizer(ngram_range=(1, n))
	return np.array(vectorizer.fit_transform(list).todense())


# In[6]:


def split_data(X, y, train=10000, test=12500):
	train_X = X[0: train, :]
	train_y = y[0: train]
	test_X = X[train: test, :]
	test_y = y[train: test]

	print("Done Spliting")
	return train_X, train_y, test_X, test_y


# In[7]:


def evaluate(y_test, y_pred):
	precision = metrics.precision_score(y_test, y_pred, average=None)
	recall = metrics.recall_score(y_test, y_pred, average=None)
	fscore = metrics.f1_score(y_test, y_pred, average=None)
	accuracy = metrics.accuracy_score(y_test, y_pred)

	print("Precision, Recall, F1 score, Accuracy")
	print(precision, recall, fscore, accuracy)

	return

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def run(X, y):
	train_X, train_y, test_X, test_y = split_data(X, y)
	nb = MultinomialNB()
	nb.fit(train_X, train_y)
	print('Done Training')

	y_pred = nb.predict(test_X)
	evaluate(test_y, y_pred)

	class_names = ['80-84', '84-88', '88-92', '92-96', '96-100']
	plot_confusion_matrix(test_y, y_pred, classes=class_names, normalize=True, title='Confusion matrix')

	plt.show()


def main(args):
	files = [args.train, args.test]
	description_list, y = read_data(files)

	# tf_bigram_X = vectorize(description_list, True, 2)

	# tf_trigram_X = vectorize(description_list, True, 3)

	tfidf_bigram_X = vectorize(description_list, False, 3)

	# tfidf_unigram_X = vectorize(description_list, False, 1)

	run(tfidf_bigram_X, y)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
