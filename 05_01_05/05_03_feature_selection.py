# 特徴量選択
import numpy as np

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size = 0.2,
                  random_state=0)
train_index,test_index = next(ss.split(X,y))
X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

data.feature_names
# 上位からK個の特徴量を選択
# 階２条基準で選択
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# 20個の特徴量を選択する分類器を作成
skb = SelectKBest(chi2,k=20)

skb.fit(X_train,y_train)

X_train_new = skb.transform(X_train)

X_train_new.shape
X_train.shape
# 利用した特徴量を表示
skb.get_support()
# 利用した特徴量は
data.feature_names[skb.get_support()]
# 利用していない特徴量は
data.feature_names[~skb.get_support()]

from sklearn import linear_model
clf = linear_model.LogisticRegression()
from sklearn.model_selection import StratifiedKFold

k_range = np.arange(1,31)
scores = []
std = []

for k in k_range:

    ss = StratifiedKFold(n_splits=10,
                   shuffle=True,
                   random_state=2)
    score = []
    for train_index,val_index in ss.split(X_train,
                                    y_train):

        X_train2,X_val = X[train_index],X[val_index]
        y_train2,y_val = y[train_index],y[val_index]

        skb = SelectKBest(chi2,k=k)
        skb.fit(X_train2,y_train2)

        X_new_train2 = skb.transform(X_train2)
        X_new_val = skb.transform(X_val)

        clf.fit(X_new_train2,y_train2)
        score.append(clf.score(X_new_val,y_val))

    scores.append(np.array(score).mean())
    std.append(np.array(score).std())

scores
scores = np.array(scores)
std = np.array(std)
scores
std

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(k_range,scores)
plt.errorbar(k_range,scores,yerr=std)
plt.ylabel("accuracy")
plt.bar(k_range,1-scores,yerr = [np.zeros(std.shape),std])
plt.ylabel("error rate")

best_k = k_range[np.argmax(scores)]
best_k

skb = SelectKBest(chi2,k=best_k)

skb.fit(X_train,y_train)
X_train_best = skb.transform(X_train)
X_test_best = skb.transform(X_test)

clf.fit(X_train_best,y_train)
clf.score(X_train_best,y_train)
clf.score(X_test_best,y_test)

clf.fit(X_train,y_train)
clf.score(X_train,y_train)
clf.score(X_test,y_test)
