import csv

import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def calculate_accuracy(dataset):
    X = dataset[:, 1:-1]
    y = dataset[:, -1]
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.3)
    clf = linear_model.LassoCV()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    success = 0
    for i in range(len(X_test)):
        if abs(y_test[i] - predicted[i]) < 1:
            success += 1
    return success / len(X_test) * 100


dataset = np.genfromtxt('winequalityN.csv', delimiter=',', skip_header=True)

with open('winequalityN.csv') as datafile:
    next(datafile)
    datareader = csv.reader(datafile, delimiter=',')
    first_col = []
    whites = 0
    for row in datareader:
        first_col.append(row[0])
        if row[0] == 'white':
            whites += 1

le = preprocessing.LabelEncoder()
first_col = np.array([le.fit_transform(first_col)]).T


dataset = np.hstack((first_col, dataset))
imp = SimpleImputer()
imp.fit(dataset)
dataset = imp.transform(dataset)
for i in range(len(dataset[0]) - 1):
    dataset[..., i] = preprocessing.normalize([dataset[..., i]])

print('All wines')
total = 0
for _ in range(10):
    acc = calculate_accuracy(dataset)
    print('accuracy =', acc)
    total += acc
print('Medium accuracy =', total / 10)

print('\nWhite wines')
total = 0
for _ in range(10):
    acc = calculate_accuracy(dataset[:whites])
    print('accuracy =', acc)
    total += acc
print('Medium accuracy =', total / 10)

print('\nRed wines')
total = 0
for _ in range(10):
    acc = calculate_accuracy(dataset[whites:])
    print('accuracy =', acc)
    total += acc
print('Medium accuracy =', total / 10)
