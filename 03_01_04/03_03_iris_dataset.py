"""
# TODO_01:iriのデータをインポート
# TODO_02:ロジスティック回帰(線形回帰)のモデルを作成
# TODO_03:訓練データとテストデータに分割
# TODO_04:訓練データで学習
# TODO_05:テストデータで識別器の性能を確認
# TODO_06:結果をwrongとcorrectに分類
"""
import numpy as np
from sklearn.datasets import load_iris
# TODO_01:
data = load_iris()
dir(data) #データ内のディレクトリを表示
X = data.data
X.shape # 説明変数は150,4次元のデータ
data.feature_names # 4次元の特徴量は
y = data.target
y.shape # 目的変数
y[:]
data.target_names # 目的変数4つのクラス
print(data.DESCR)
# TODO_02
from sklearn import linear_model

clf = linear_model.LogisticRegression()
clf

# TODO_03
n_samples = X.shape[0] #目的変数の数を代入
n_train = n_samples//2 #目的変数の数を2つに分割
n_test = n_samples - n_train # テストデータの始めの数字を代入
n_test
train_index = range(0,n_train)# 訓練データ用のindexを作成
test_index = range(n_train,n_samples)# テストデータ用のindexを作成

np.array(train_index),np.array(test_index)
X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

# TODO_04
clf.fit(X_train,y_train)

# TODO_05
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))

# TODO_06
clf.predict(X_test),y_test
wrong = 0
for i,j in zip(clf.predict(X_test),y_test):
    if i==j:
        print(i,j,"Correct")
    else:
        print(i,j,"Wrong!")
        wrong +=1
wrong
print("wrong:{}/n-test:{}=Correct:{}".format(wrong,n_test,1-wrong/n_test))
"""
# DONE:訓練用データとテスト用データの分割方法が悪い為、
       汎化性能が低い!
"""
"""
# TODO_00:汎化性能を高める為、訓練用データとテスト用データの分割方法を変更
# TODO_01:sklearn.model_selectionからShuffleSplitをインポート
# TODO_02:ShuffleSplitのパラメータを設定
# TODO_03:ロジスティック回帰(線形回帰)のモデルを作成
# TODO_04:ShuffleSplitで訓練データとテストデータに分割
# TODO_05:訓練データで学習
# TODO_06:テストデータで識別器の性能を確認
# TODO_07:結果をwrongとcorrectに分類
"""
# TODO_01
from sklearn.model_selection import ShuffleSplit
# TODO_02
ss = ShuffleSplit(n_splits = 1,#分割を1個作成
                  train_size = 0.5,#学習は半分
                  test_size=0.5,#テストも半分
                  random_state=0)#乱数種(再現用)
ss
# TODO_03
from sklearn.linear_model import LogisticRegression
# TODO_04
train_index,test_index = next(ss.split(X)) #ShuffleSplitで訓練用データとテストデータを分割するindexを作成
list(train_index)
len(train_index)
list(test_index)
X_train,X_test = X[train_index],X[test_index]#作成したindexを説明変数に適用
y_train,y_test = y[train_index],y[test_index]#作成したindexを目的変数にも適用
y_train,y_test
# TODO_05
clf.fit(X_train,y_train)
# TODO_06
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
clf.predict(X_test),y_test
# TODO_07
wrong = 0
for i,j in zip(clf.predict(X_test),y_test):
    if i==j:
        print(i,j,"Correct")
    else:
        print(i,j,"Wrong!")
        wrong +=1
wrong
n_test
1-wrong/n_test
print("{}/{}={}".format(wrong,n_test,1-wrong/n_test))
"""
# DONE:訓練用データとテストデータの分割方法を変更した結果汎化性能が大幅にUP
"""
