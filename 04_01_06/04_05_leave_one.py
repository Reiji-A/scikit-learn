import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target
X.size
y.size

from sklearn import linear_model
clf = linear_model.LogisticRegression()

from sklearn.model_selection import LeaveOneOut
loocv = LeaveOneOut()

train_index,test_index = next(loocv.split(X,y))
y.size,train_index.size,test_index.size,

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,
                     X,y,
                     cv=loocv)#LeaveOneOut

scores.mean()*100
scores.std()*100

# leave p(2) outは終わらない(組み合わせが大量にある為)
# LeaveOneOutを使う
from sklearn.model_selection import LeavePOut
loocv = LeavePOut(2)

# leave one-group out
group = np.array(list(range(50))*12)
group = np.sort(group[:y.size])
group.size
group
from sklearn.model_selection import LeaveOneGroupOut
loocv = LeaveOneGroupOut()

for train_index, test_index in loocv.split(X, y, group):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,
                X,y,
                groups=group,
                cv=loocv)
scores.mean()
scores.std()
scores.size
