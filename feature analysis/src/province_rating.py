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
  label = ['Chardonnay', 'Pinot Noir', 'Cabernet Sauvignon', 'Red Blend', 'Sauvignon Blanc']
  # label = ['California', 'Washington', 'NorthernSpain', 'Bordeaux', 'Tuscany']
  prices = []
  for index, hit in data.iterrows():
    # p = label.index(hit['province'])
    p = label.index(hit['variety'])
    X[i, :] = np.array(p)
    points = int(hit['points'])
    Y.append(points)
    i += 1
  return X, Y


if __name__ == '__main__':
	data = pd.read_csv('train3.csv')

	X_train, Y_train = test(44349, data)
	print('Retrieved data')
	model = LogisticRegression().fit(X_train, Y_train)
	print('Training complete')

	datat = pd.read_csv('test3.csv')
	X_test, Y_test = test(5577, datat)
	Y_pred = model.predict(X_test)

	print('F1: {}'.format(f1_score(Y_test, Y_pred, average='micro')))
	print(Y_pred)

	with open('test_output_province.txt' ,'w') as f:
	  for i in Y_pred:
	    f.write(str(i) + '\n')