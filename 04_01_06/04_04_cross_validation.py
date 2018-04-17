"""
TODO_00:学習データでの汎化性能を確かめる為、cross_validation法を用いる
TODO_01:ガンのデータをインポート
TODO_02:KFoldで10つのサイズにデータを分割させLogisticRegressionで学習させるモデルを作成
TODO_03:KFoldのモデルを利用して訓練用、学習用のデータを分割するindexを作成し生データに適用
TODO_04:ロジスティック回帰で10回訓練データで学習させる
TODO_05:10回分のスコアを確認
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target

from sklearn import linear_model
clf = linear_model.LogisticRegression()

from sklearn.model_selection import KFold
ss = KFold(n_splits=10,shuffle=True)
ss
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
print(np.unique(y_train,return_counts = True)[1]/y_train.size)
"""
DONE:訓練データの比率がテストデータのtargetの比率が同じ、汎化性能が高いかも?
"""
ss = KFold(n_splits=10,shuffle=True)

for train_index,test_index in ss.split(X,y):

    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    print(np.unique(y,return_counts = True)[1]/y.size,y.size,
        np.unique(y_train,return_counts = True)[1]/y_train.size,y_train.size,
        np.unique(y_test,return_counts = True)[1]/y_test.size,y_test.size)
"""
DONE:層化されていない為、サイズがバラバラになっている、汎化性能が高いとは言えない
"""
"""
TODO_00:StratifiedKFoldで層化したクロスバリデーションを行う
"""
from sklearn.model_selection import StratifiedKFold
ss = StratifiedKFold(n_splits=10,shuffle=True)

for train_index,test_index in ss.split(X,y):

    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    print(np.unique(y_train,return_counts = True)[1]/y_train.size,y_train.size,
        np.unique(y_test,return_counts = True)[1]/y_test.size,y_test.size)
"""
DONE:層化されている為、サイズがほぼ一定、汎化性能が高い
"""
# crossvalidationでstratified(クラスを階層ごとに分割)し、スコアをave_scoreに格納するモデル
"""
TODO_00:cross_val_scoreを実施,LogisticRegressionで学習
TODO_01:ave_scoreにスコアが格納されている為表示
"""
from sklearn.model_selection import cross_val_score
ave_score = cross_val_score(clf,
                  X,y,
                  cv=10)

ave_score
ave_score.mean()*100
ave_score.std()*100
print("{:4.2f} +/- {:4.2f}".format(ave_score.mean()*100,ave_score.std()*100))
