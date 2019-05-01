# cis530_wine_project
The objective of this project is to predict wine rating based on the review

## Data
Data folder contains train, val, and test data that are preprocessed. 
Stop words and punctuations are removed, and words are lemmatized.

## Baseline
Baseline folder contains source code of baseline model. 
_length.py_ computes the accuracy based by training a linear model using the length of wine reviews

## Wordvec
Wordvec folder contains source code for 2 wordvec models:
1) _concat.py_: concatenates the word vector of words in each wine review to train model.
2) _average.py_: averages the word vector of words in each wine review to train model.
