import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target

from sklearn import linear_model
clf = linear_model.LogisticRegression()

from sklearn.model_selection import KFold
ss = KFold(n_splits=10,shuffle=True)

scores = []
for train_index,test_index in ss.split(X,y):

    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    scores.append(score)
scores
print(np.unique(y,return_counts = True))
print(np.unique(y,return_counts = True)[1]/y.size)

ss = KFold(n_splits=10,shuffle=True)

for train_index,test_index in ss.split(X,y):

    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    print(np.unique(y_train,return_counts = True)[1]/y_train.size,y_train.size,
        np.unique(y_test,return_counts = True)[1]/y_test.size,y_test.size)

from sklearn.model_selection import StratifiedKFold
ss = StratifiedKFold(n_splits=10,shuffle=True)

for train_index,test_index in ss.split(X,y):

    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    print(np.unique(y_train,return_counts = True)[1]/y_train.size,y_train.size,
        np.unique(y_test,return_counts = True)[1]/y_test.size,y_test.size)
# crossvalidationでstratified(クラスを階層ごとに分割)し、スコアをave_scoreに格納
from sklearn.model_selection import cross_val_score
ave_score = cross_val_score(clf,
                  X,y,
                  cv=10)
ave_score
ave_score.mean()*100
ave_score.std()*100
print("{:4.2f} +/- {:4.2f}".format(ave_score.mean()*100,ave_score.std()*100))
