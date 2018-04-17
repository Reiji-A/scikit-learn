import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
X.shape
X[0]
data.feature_names
y = data.target
y.shape
y[0]
data.target_names
print(data.DESCR)
#線形モデル
from sklearn import linear_model

# ロジスティック回帰(線形回帰)の識別器を作成
clf = linear_model.LogisticRegression()
clf
#学習データとテストデータを分割
n_samples = X.shape[0]
n_train = n_samples//2
n_test = n_samples - n_train

train_index = range(0,n_train)
test_index = range(n_train,n_samples)

np.array(train_index),np.array(test_index)
X_train = X[train_index]
X_test = X[test_index]

y_train = y[train_index]
y_test = y[test_index]

# 識別器に学習
clf.fit(X_train,y_train)

# 学習データのテスト
print(clf.score(X_train,y_train))
# テストデータのテスト
print(clf.score(X_test,y_test))
# テストデータの識別結果
clf.predict(X_test)
# y_testと上記識別結果を比較
wrong = 0
for i,j in zip(clf.predict(X_test),y_test):
    if i == j:
        print(i,j)
    else:
        print(i,j,"Wrong")
        wrong +=1
1-wrong/n_test
print("{}/{}={}".format(wrong,n_test,1-wrong/n_test))
