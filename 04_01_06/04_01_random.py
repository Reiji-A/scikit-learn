"""
# TODO_00: 説明変数として0~1までの適当な数字を500行2列のデータを作成
           目的変数として0と1の2クラスの500のデータを作成する、
           説明変数から正確に2クラス分割する汎化性能の高いモデルを探す
# TODO_01:numpy.random.uniformで500行2列の説明変数を作成
# TODO_02:numpy.random.choiceで500行の目的変数を作成
# TODO_03:説明変数を目的変数で色分けしたグラフをmatplotlibで描画
# TODO_04:k-近傍法で訓練データとテストデータを分割せずに学習させる
# TODO_05:別の500行2列の説明変数を用意しTODO_04のモデルで再度テストしてみる
"""

import numpy as np
N = 500
# TODO_01
X = np.random.uniform(low = 0,high=1,size=[N,2])
X
# TODO_02
y= np.random.choice([0,1],size=N)
y
#TODO_03
import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap=plt.cm.Paired,edgecolors="k")
from sklearn import neighbors
# TODO_04
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
# 学習データとテストデータを分割
X_train = X
X_test = X

y_train = y
y_test = y
#モデルを作成
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# TODO_05
X_test2 = np.random.uniform(low = 0,high=1,size=[N,2])
X_test2.shape
# 出力データとして0か1の出力データを500を作成
y_test2= np.random.choice([0,1],size=N)
y_test2.shape
clf.score(X_test2,y_test2)
"""
# DONE:学習データとテストデータを分割せずに学習させると汎化性能が高くないモデルが作成される
"""
