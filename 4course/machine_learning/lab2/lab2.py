# Обработка данных
from itertools import product
import numpy as np
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
dataset = np.genfromtxt('heart_data.csv', delimiter=',', skip_header=True)
X = dataset[:, :-1]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.3)
imp = SimpleImputer()
imp.fit(X_train)
X_train = imp.transform(X_train)

for max_depth, max_leaf_nodes in product(range(10, 100, 10), range(10, 100, 10)):
    # Обучение
    clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    clf.fit(X_train, y_train)

    # Проверка
    X_test = imp.transform(X_test)
    predicted = clf.predict(X_test)
    print(f'{sk.metrics.accuracy_score(y_test, predicted)} ({max_depth = }, {max_leaf_nodes = })')
