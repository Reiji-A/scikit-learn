"""
TODO_00:学習データの汎化性能を確かめる為、holdout法を用いる
TODO_01:ガンのデータをインポート
TODO_02:ShuffleSplitで1つの訓練サイズ0.5、テストサイズ0.5で学習させるモデルを作成
TODO_03:ShuffleSplitのモデルを利用して訓練用、学習用のデータを分割するindexを作成し生データに適用
TODO_04:ロジスティック回帰に訓練データで学習させる
TODO_05:スコアを確認
TODO_06:ShuffleSplitで10つの訓練:0.5,テスト:0.5,訓練:0.95,テスト:0.05で学習とスコアをfor文で回しスコアを確認
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
# TODO_01
data = load_breast_cancer()

dir(data)
X = data.data
y = data.target

from sklearn import linear_model
clf = linear_model.LogisticRegression()
# TODO_02
from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1,
                  train_size=0.5,
                  test_size=0.5)
# TODO_03
train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]
# TODO_04
clf.fit(X_train,y_train)
# TODO_05
clf.score(X_test,y_test)

# TODO_06
ss = ShuffleSplit(n_splits=10,
                  train_size=0.5,
                  test_size=0.5)
ss
scores = []
for train_index,test_index in ss.split(X,y):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    scores.append(score)

scores
"""
DONE:大きな汎化性能の違いわ見られない
"""
ss = ShuffleSplit(n_splits=10,
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
"""
DONE:訓練データの比率がテストデータのtargetの比率が大きく違う為、汎化性能が高いとは言えない
"""
scores = []
for train_index,test_index in ss.split(X,y):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    scores.append(score)

scores
"""
DONE_02:汎化性能が大きく違う
"""
"""
TODO_00:ShuffleSplitではなく、StratifiedShuffleSplitで訓練データとテストデータのさいぞを変える
"""

from sklearn.model_selection import StratifiedShuffleSplit

ss = StratifiedShuffleSplit(n_splits=10,
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
"""
DONE:訓練データの比率がテストデータのtargetの比率が同じ、汎化性能が高いと言えるかも
"""
scores = []
for train_index,test_index in ss.split(X,y):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    scores.append(score)

scores

"""
DONE : 学習データとテストデータが同じサイズに保たれている(層化)
"""

clf.fit(X_train,y_train)
clf.score(X_test,y_test)
