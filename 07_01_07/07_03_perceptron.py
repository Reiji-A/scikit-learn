import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=20,
                 n_features=2,
                 centers=2,
                 cluster_std=2,
                 random_state=3)

import matplotlib.pyplot as plt
%matplotlib inline
plt.set_cmap(plt.cm.Paired)

plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="k")

# 境界線を引く関数の定義
def plotBoundary(X, clf, mesh=True, boundary=True, n_neighbors=1):

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid

    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()]) # evaluate the value

    Z = Z.reshape(XX.shape) # just reshape

    if mesh:
        plt.pcolormesh(XX, YY, Z, zorder=-10) # paint in 2 colors, if Z > 0 or not

    if boundary:
        plt.contour(XX, YY, Z,
                    colors='k', linestyles='-', levels=[0])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

from sklearn.linear_model import Perceptron

clf = Perceptron()
clf

clf.fit(X,y)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
plotBoundary(X, clf) # 境界線の描画

from IPython.display import HTML
HTML('<iframe scrolling="no" src="https://www.geogebra.org/material/iframe/id/K8kugARA/width/939/height/469/border/888888/sri/true/sdz/true" width="939px" height="469px" style="border:0px;"> </iframe>')

X, y = make_blobs(n_samples=20, # 20個生成
                  n_features=2, # 2次元
                  centers=2,    # クラスタ中心2個
                  cluster_std =1, # クラスタの大きさ（標準偏差）
                  random_state=8   # 乱数種（再現用）
                 )
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k');

for s in range(10):
    clf.random_state = s
    clf.fit(X,y)

    plotBoundary(X,clf,mesh=False)

plt.scatter(X[:,0],X[:,1],marker="o",s=50,c=y,edgecolors="k")

clf.coef_,clf.intercept_

X, y = make_blobs(n_samples=20, # 20個生成
                  n_features=2, # 2次元
                  centers=2,    # クラスタ中心2個
                  cluster_std =1, # クラスタの大きさ（標準偏差）
                  random_state=7   # 乱数種（再現用）
                 )
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k');

clf.warm_start = True

for s in range(10):

    clf.random_state = s

    clf.coef_, clf.intercept_ = np.random.rand(1,2) * 10 - 5, np.random.rand(1) * 10 + 30

    clf.fit(X,y)

    plotBoundary(X, clf, mesh=False)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

X, y = make_blobs(n_samples=20, # 20個生成
                  n_features=2, # 2次元
                  centers=2,    # クラスタ中心2個
                  cluster_std =1, # クラスタの大きさ（標準偏差）
                  random_state=7   # 乱数種（再現用）
                 )
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k');

clf.warm_start = False

for s in range(10):

    clf.random_state = s

    clf.fit(X,y)

    plotBoundary(X, clf, mesh=False)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

clf = Perceptron()
clf.random_state = 4

for rs in [8,7,4,5]:

    X, y = make_blobs(n_samples=20, # 20個生成
                      n_features=2, # 2次元
                      centers=2,    # クラスタ中心2個
                      cluster_std =1, # クラスタの大きさ（標準偏差）
                      random_state=rs
                     )
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k');


    for i in range(1,20):

        clf.n_iter = i # epochs

        clf.fit(X,y)

        plotBoundary(X, clf, mesh=False)

    plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
    plt.show()


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size=0.2,
                  random_state=0)

train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

clf = Perceptron()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

clf = Perceptron(warm_start=True,n_iter=1)

scores = []
n_range = range(1,50)
for n in n_range:
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    print(n,score)
    scores.append(score)
scores = np.array(scores)

plt.plot(n_range,scores)
plt.xlabel("epochs")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

clf = Perceptron(warm_start=True,n_iter=1)

scores2 = []
n_range = range(1,50)
for n in n_range:
    clf.fit(X_train_scale,y_train)
    score = clf.score(X_test_scale,y_test)
    print(n,score)
    scores2.append(score)
scores2 = np.array(scores2)

plt.plot(n_range,scores,label="no scaling")
plt.plot(n_range,scores2,label="scaling")
plt.legend(loc = "best")
plt.xlabel("epochs")

clf = Perceptron(n_iter=50)
clf.fit(X_train_scale,y_train)
clf.score(X_test_scale,y_test)

clf = Perceptron()#default n_iter=5
clf.fit(X_train,y_train)#no scaling
clf.score(X_test,y_test)
