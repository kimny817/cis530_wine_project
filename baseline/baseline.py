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
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, required=True)
parser.add_argument('--test', type=str, required=True)

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def classify(rating):
	# Assuming that 80 <= rating <=100
	return min(5, int((rating-80)/4) + 1)


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
	vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')

	description_list = []
	labels = []

	with open(file_path, 'r') as f:
		csv_reader = csv.reader(f, delimiter=',')
		next(csv_reader) # This skips the first row of the CSV file.

		for row in csv_reader:
			description = row[0]
			rating = classify(int(row[1]))
			# tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', description).split()
			# tokens = [t.lower() for t in tokens]
			# tokens = [lemmatizer(w) for w in tokens if w not in stopwords_set]
			labels.append(rating)
			description_list.append(description)

	# Cannot fit the test data
	#if is_train:
	return vectorizer.fit_transform(description_list).todense(), labels
	#else:
	#	return vectorizer.transform(description_list), labels


def baseline(train_file, test_file):
	# Change train_file to entire file
	X, y = read_data(train_file, True)
	np_X = np.array(X)
	np_y = np.array(y)

	train_X, test_X, train_y, test_y = train_test_split(np_X, np_y, test_size=0.1, random_state=69)

	gnb = MultinomialNB()
	gnb.fit(train_X, train_y)

	# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

	y_pred = gnb.predict(test_X)

	precision = metrics.precision_score(test_y, y_pred, average = None)
	recall = metrics.recall_score(test_y, y_pred, average = None)
	fscore = metrics.f1_score(test_y, y_pred, average = None)
	accuracy = metrics.accuracy_score(test_y, y_pred)

	print("Precision, Recall, F1 score, Accuracy")
	print(precision, recall, fscore, accuracy)


def main(args):
	baseline(args.train, args.test)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)