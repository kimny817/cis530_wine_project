'''
File name: predict_variety.py
Description:
'''

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

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def read_data(file):
	stopwords_set = set(stopwords.words('english'))
	description_list = []
	labels = []
	top_varieties = ['Chardonnay', 'Pinot Noir', 'Cabernet Sauvignon', 'Red Blend', 'Bordeaux-style Red Blend']

	with open(file, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',')
		next(csv_reader)  # This skips the first row of the CSV file.

		for row in csv_reader:
			description = row[0]
			variety = row[1]

			if variety in top_varieties:
				tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', description).split()
				tokens = [t.lower() for t in tokens]
				tokens = [lemmatizer(w) for w in tokens if w not in stopwords_set]
				labels.append(variety)
				description_list.append(description)

	return description_list, np.array(labels)

def split_data(X, y, train=10000, test=12500):
	train_X = X[0: train, :]
	train_y = y[0: train]
	test_X = X[train: test, :]
	test_y = y[train: test]

	print("Done Spliting")
	return train_X, train_y, test_X, test_y

def vectorize(list, is_ngram, n):
	vectorizer = CountVectorizer(ngram_range=(1, n)) if is_ngram else TfidfVectorizer(ngram_range=(1, n))
	return np.array(vectorizer.fit_transform(list).todense())

def evaluate(y_test, y_pred):
	precision = metrics.precision_score(y_test, y_pred, average=None)
	recall = metrics.recall_score(y_test, y_pred, average=None)
	fscore = metrics.f1_score(y_test, y_pred, average=None)
	accuracy = metrics.accuracy_score(y_test, y_pred)

	print("Precision, Recall, F1 score, Accuracy")
	print(precision, recall, fscore, accuracy)

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

	class_names = ['California', 'Washington', 'Tuscany', 'Bordeaux', 'Northern Spain']
	plot_confusion_matrix(test_y, y_pred, classes=class_names, normalize=True, title='Confusion matrix')

	plt.show()

if __name__ == '__main__':
	reviews, variety = read_data('wine_top_variety.csv')
	tfidf_bigram_X = vectorize(reviews, False, 3)
	run(tfidf_bigram_X, variety)