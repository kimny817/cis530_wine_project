import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

embedding_dim = 300
vocab_vec = {}


def createVector(file):
  n = sum(1 for line in open(file))
  X = np.zeros((n, 1))
  Y = []
  lengths = []
  i = 0  
  with open(file, 'r') as f:
    for line in f:
      words = line[:len(line) - 3].split()
      label = line[len(line) - 3: len(line) - 2]
      if label[0] == '1' or label[0] == '2' or label[0] == '3' or label[0] == '4' or label[0] == '5':
        X[i, :] = len(words)
        lengths.append(len(words))
        Y.append(int(label))
        i += 1
  # return X, Y
  return lengths, Y

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
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

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

if __name__ == '__main__':
  X_train, Y_train = createVector('train.csv')
  print('Vector Created')
  model = LogisticRegression().fit(X_train[:len(X_train) - 1], Y_train)
  print('Training complete')
  print('Correlation: ' + str(pearsonr(X_train, Y_train)[0]))

  X_test, Y_test = createVector('test.csv')
  Y_pred = model.predict(X_test)

  print('F1: {}'.format(f1_score(Y_test, Y_pred, average='micro')))

  with open('test_output_len.txt' ,'w') as f:
    for i in Y_pred:
      f.write(i + '\n')

  # class_names = ['80-84', '84-88', '88-92', '92-96', '96-100']
  # plot_confusion_matrix(Y_test, Y_pred, classes=class_names, normalize=True, title='Confusion Matrix')