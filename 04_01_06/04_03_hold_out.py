import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

dir(data)
X = data.data
y = data.target

from sklearn import linear_model
clf = linear_model.LogisticRegression()

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1,
                  train_size=0.5,
                  test_size=0.5)

train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# 分割する数を10に変更し10回学習を実施
ss = ShuffleSplit(n_splits=10,
                  train_size=0.5,
                  test_size=0.5)

scores = []
for train_index,test_index in ss.split(X,y):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    scores.append(score)

scores

ss = ShuffleSplit(n_splits=1,
                  train_size=0.95,
                  test_size=0.05,
                  random_state=3)

train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]
# yの要素を表示、要素の数を表示
np.unique(y,return_counts = True)
# 要素の数をyの要素数で割り比率を表示
y.size
np.unique(y,return_counts = True)[1]/y.size
np.unique(y_train,return_counts = True)[1]/y_train.size
np.unique(y_test,return_counts = True)[1]/y_test.size

from sklearn.model_selection import StratifiedShuffleSplit

ss = StratifiedShuffleSplit(n_splits=1,
                  train_size=0.95,
                  test_size=0.05,
                  random_state=3)

train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

print(np.unique(y,       return_counts=True))
print(np.unique(y,       return_counts=True)[1] / y.size)
print(np.unique(y_train, return_counts=True)[1] / y_train.size)
print(np.unique(y_test,  return_counts=True)[1] / y_test.size)
