"""
# TODO_00:汎化性能を高める為,訓練データとテストデータの分割方法を工夫してみよう
# TODO_01:データとShuffleSplit,ロジスティック回帰(線形回帰)をインポート
# TODO_02:ShuffleSplitのパラメータを設定
# TODO_03:ロジスティック回帰(線形回帰)のモデルを作成
# TODO_04:ShuffleSplitで訓練データとテストデータに分割
# TODO_05:訓練データで学習
# TODO_06:テストデータで識別器の性能を確認
"""
# TODO_01
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model

data = load_iris()
X = data.data
y = data.target
# TODO_02
ss = ShuffleSplit(n_splits=10,#分割を10個作成
                  train_size=0.5,#訓練サイズを0.5
                  test_size=0.5,#テストサイズを0.5
                  random_state=0)#再現用に乱数を規定
ss
# TODO_03
clf = linear_model.LogisticRegression()
clf
# TODO_04~06
scores = []
for train_index,test_index in ss.split(X):#TODO_04
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]

    clf.fit(X_train,y_train)#TODO_05
    score = clf.score(X_test,y_test)#TODO_06
    scores.append(score)

scores = np.array(scores)
scores.shape
print(scores)
scores.mean()
scores.std()
print("{:0.2f}%+/-{:.4f}%".format(scores.mean(),scores.std()))
"""
# DONE:分割を10回繰り返すと汎化性能に違いが出ることがわかった
"""
"""
# TODO_00:より汎化性能を高める為、
          分割方法を1/2ではなく、訓練データとテストデータのサイズを色々変化させてみる
# TODO_01:numpyのarange関数で0.1から1までを0.1刻みのアレイを作成
# TODO_02:ShuffleSplitのパラメータを設定トレインサイズとテストサイズを変化させるfor文を作成
# TODO_03:ロジスティック回帰(線形回帰)のモデルを作成
# TODO_04:ShuffleSplitで訓練データとテストデータに分割
# TODO_05:訓練データで学習
# TODO_06:テストデータで識別器の性能を確認
"""

import matplotlib.pyplot as plt
%matplotlib inline
# TODO_01
train_sizes = np.arange(0.1,1.0,0.1)# 0.1から0.9までを0.1刻みで作成
train_sizes
# 平均と標準偏差用のリストを準備
all_mean =[]
all_std = []
# TODO_01~TODO_05
for train_size in train_sizes:

    ss = ShuffleSplit(n_splits=100,
                      train_size=train_size,
                      test_size=1-train_size)

    scores = []
    for train_index, test_index in ss.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)

    scores = np.array(scores)
    print("train_size{0:.0f}%:{1:4.2f} + / - {2:4.2f}%".format(train_size*100,
                                         scores.mean()*100,scores.std()*100))
    all_mean.append(scores.mean()*100)
    all_std.append(scores.std()*100)
"""
# DONE:訓練サイズが80%の時が汎化性能が最も高く、標準偏差が-3.83となっている
# TODO_01:訓練サイズを変化させたテスト結果を描画
"""
# TODO_01
plt.plot(train_sizes,all_mean)

plt.plot(train_sizes,all_mean)
plt.ylim(70,100)
plt.xlim(0,1)
plt.plot(train_sizes,all_mean)
plt.ylim(70,100)
plt.xlim(0,1)
plt.errorbar(train_sizes,all_mean,yerr=all_std)
plt.xlabel("training size [%]")
plt.ylabel("recognition rate")
plt.title("Average of 10 hold-out tests for different training-size")
