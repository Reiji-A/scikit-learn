import numpy as np
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=50,#20個生成
           n_features=2,#2次元
           centers=5,#クラスタ中心2個
           cluster_std=.8,#クラスタの大きさ(標準偏差)
           random_state=3#乱数種(再現用)
           )
X.shape,y

import matplotlib.pyplot as plt
%matplotlib inline
plt.set_cmap(plt.cm.gist_ncar)

plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="k")
plt.show()

#境界線を引く関数の定義
def plotBoundary(X, clf, mesh=True, cmap=plt.get_cmap()):

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
        plt.pcolormesh(XX, YY, Z, zorder=-10, cmap=cmap)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c = y,s=50,edgecolors="k")
plotBoundary(X,clf)
plt.show()

# 別のデータを生成
X, y = make_blobs(n_samples=50, # 20個生成
                  n_features=2, # 2次元
                  centers=5,    # クラスタ中心2個
                  cluster_std =2, # クラスタの大きさ（標準偏差）
                  random_state=3   # 乱数種（再現用）
                 )
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k');
plt.show()

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')

plotBoundary(X, clf) # 境界線の描画
plt.show()

for n in [1,5,10,15]:
    clf.n_neighbors = n
    plt.scatter(X[:,0],X[:,1],marker="o",s=50,c=y,edgecolors="k")
    plotBoundary(X,clf)
    plt.title("{}-NN".format(n))
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

from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,y_train)

clf.score(X_train,y_train)
clf.score(X_test,y_test)

n_range = range(1,20)
scores = []
for i in n_range:
    clf.n_neighbors = i
    score = clf.score(X_test,y_test)
    print(i,score)
    scores.append(score)
scoers = np.array(scores)
len(scores)
plt.plot(n_range,scoers)
plt.ylim(0.8,1)
plt.show()

clf = neighbors.RadiusNeighborsClassifier()
clf.fit(X_train,y_train)
n_range = [2000,4000,8000]
for i in n_range:
    clf.radius = i
    score= clf.score(X_test,y_test)
    print(i,score)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
clf.fit(X_train_scale,y_train)
X_test_scale = scaler.transform(X_test)

n_range = [3,4,5,6,7]
for i in n_range:
    clf.radius = i
    score = clf.score(X_test_scale,y_test)
    print(i,score)


clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_scale,y_train)
n_range = range(1,20)
scores2 = []
for n in n_range:
    clf.n_neighbors = n
    score = clf.score(X_test_scale,y_test)
    print(n,score)
    scores2.append(score)
scores2 = np.array(scores2)

plt.plot(n_range,scores2,label="scailing")
plt.plot(n_range,scores,label="no scaling")
plt.legend()
plt.ylim(0.8,1)
plt.show()
