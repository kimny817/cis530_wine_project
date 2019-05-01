import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

def test(n, data):
  X = np.zeros((n, 1))
  Y = []
  i = 0  
  prices = []
  for index, hit in data.iterrows():
    price = int(hit['price'])
    prices.append(price)
    X[i, :] = np.array(price)
    points = int(hit['points'])
    Y.append(points)
    i += 1
  return X, Y, prices


if __name__ == '__main__':
	data = pd.read_csv('train.csv')

	X_train, Y_train, p = test(109789, data)
	print('Retrieved data')
	print('Correlation: ' + str(pearsonr(p, Y_train)[0]))
	model = LogisticRegression().fit(X_train, Y_train)
	print('Training complete')

	datat = pd.read_csv('test.csv')
	X_test, Y_test, p = test(13707, datat)
	Y_pred = model.predict(X_test)

	print('F1: {}'.format(f1_score(Y_test, Y_pred, average='micro')))
	print(Y_pred)

	with open('test_output_price.txt' ,'w') as f:
	  for i in Y_pred:
	    f.write(str(i) + '\n')