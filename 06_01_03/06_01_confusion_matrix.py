"""
# TODO_0:matrix表現でテストデータに正解と不正解を表示する。
        1番目はガンのデータでMaligant(悪性)とBenign(良性)の2クラス問題
        2番目は画像認識で0~9までの数字を判別する10クラス問題
"""
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

print(data.DESCR)

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits = 1,
                  train_size = 0.8,
                  test_size = 0.2,
                  random_state = 0)

train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_test
y_pred = clf.predict(X_test)
wrong=0
for i,j in zip(clf.predict(X_test),y_test):
    if i == j:
        print(i,j,"Correct!")
    else:
        print(i,j,"Wrong")
        wrong +=1
print(wrong)
conf_mat = np.zeros([2,2])
conf_mat
for true_label,est_label in zip(y_test,y_pred):
    conf_mat[true_label,est_label] +=1
conf_mat

import pandas as pd
df = pd.DataFrame(conf_mat,
                 columns=["pred 0","pred 1"],
                 index = ["true 0","true 1"])
df
len(y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_test,y_pred)

cmat = confusion_matrix(y_test,y_pred)
cmat
# True Positive 真陽性
TP = cmat[0,0]
TP
# True negative 真陰性
TN = cmat[1,1]
TN
# False Positive 偽陽性
FP = cmat[1,0]
FP
# False Negative 偽陰性
FN = cmat[0,1]
FN

# 10class problem
from sklearn.datasets import load_digits
data = load_digits()

X = data.data
y = data.target
img = data.images
X[0].shape,img[0].shape

import matplotlib.pyplot as plt
%matplotlib inline
plt.gray()
plt.imshow(img[0],interpolation=None)
plt.axis("off")
X
for i in range(10):
    i_th_digit = data.images[data.target == i]
    for j in range(0,15):
        plt.subplot(10,15,i*15+j+1)
        plt.axis("off")
        plt.imshow(i_th_digit[j],interpolation="none")
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size=0.2,
                  random_state=0)
train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat
df = pd.DataFrame(conf_mat,
                  columns=range(0,10),
                  index = range(0,10))
df
# スケーリングを実施
#PCA(主成分分析)
from sklearn.decomposition import PCA
pca = PCA(whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

for i in range(10):
    i_th_digit = X_train_pca[y_train == i]
    for j in range(0,15):
        plt.subplot(10, 15, i * 15 + j +1)
        plt.axis('off')
        plt.imshow(i_th_digit[j].reshape(8,8), interpolation='none')

clf.fit(X_train_pca,y_train)
clf.score(X_test_pca,y_test)

y_pred_pca = clf.predict(X_test_pca)
conf_mat = confusion_matrix(y_test,y_pred_pca)
df = pd.DataFrame(conf_mat,
              columns=range(0,10),
              index=range(0,10))
df

X_train_zca = X_train_pca.dot(pca.components_)
X_test_zca = X_test_pca.dot(pca.components_)

for i in range(10):
    i_th_digit = X_train_zca[y_train == i]
    for j in range(0,15):
        plt.subplot(10, 15, i * 15 + j +1)
        plt.axis('off')
        plt.imshow(i_th_digit[j].reshape(8,8), interpolation='none')


clf.fit(X_train_zca,y_train)
clf.score(X_test_zca,y_test)
y_pred_zca = clf.predict(X_test_zca)
conf_mat = confusion_matrix(y_test,y_pred_zca)
df = pd.DataFrame(conf_mat,
              columns=range(0,10),
              index=range(0,10))
df

# 次元削減
scores = []
for i in range(1,65):
    clf.fit(X_train_pca[:, 0:i], y_train)
    score = clf.score(X_test_pca[:, 0:i], y_test)
    print(i, score)
    scores.append(score)
scores = np.array(scores)
scores
plt.plot(scores)
plt.ylim(0.9,1)
