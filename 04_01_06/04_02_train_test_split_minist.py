"""
# TODO_00:ガンのデータをトレインデータとテストデータに分割しロジスティック回帰で学習させ
          テスト結果を表示
# TODO_01:ガンのデータをインポート
# TODO_02:ShuffleSplitで1つの訓練サイズ0.5、テストサイズ0.5で学習させるモデルを作成
# TODO_03:ShuffleSplitのモデルを利用して訓練用、学習用のデータを分割するindexを作成し生データに適用
# TODO_04:ロジスティック回帰に訓練データで学習させる
# TODO_05:スコアを確認
# TODO_06:文字のデータをインポート
# TODO_07:70000行784の特徴量を持つ説明変数を訓練データとテストデータに分割させ9クラスの目的変数にロジスティック回帰で学
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
X.shape
y.shape
# TODO_03
train_index,test_index = next(ss.split(X,y))
# TODO_04
clf.fit(X[train_index],y[train_index])
# TODO_05
clf.score(X[test_index],y[test_index])

# TODO_06
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")

mnist.COL_NAMES
mnist.DESCR

mnist.data.shape
mnist.target
# TODO_07
X_train = mnist.data[:60000]
X_test = mnist.data[60000:70000]

y_train = mnist.target[:60000]
y_test = mnist.target[60000:70000]

clf

# clf.fit(X_train,y_train)

"""
DONE:実際のデータに学習させるにはモデルを選ぶ必要がある
"""
