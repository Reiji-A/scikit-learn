import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
X,y = make_classification(n_samples=20,#20個生成
                          n_features=2,#2次元
                          n_classes = 3,#3クラス
                          n_clusters_per_class=1,
                          n_informative=2,
                          n_redundant=0,
                          n_repeated=0,
                          random_state=8#乱数種(再現用)
                          )
X,y
df = pd.DataFrame(X,y,columns=None)
df
import matplotlib.pyplot as plt
%matplotlib inline
plt.set_cmap(plt.cm.brg)

plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="k")

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

plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="k")
plotBoundary(X,clf)

from sklearn import svm

clf = svm.SVC(kernel='linear',C=10)
clf.fit(X,y)

plt.scatter(X[:,0],X[:,1],marker='o',c=y,s=50,edgecolors="k")
plotBoundary(X,clf)

from sklearn import svm

clf = svm.SVC(kernel='rbf',C=10)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],marker='o',c=y,s=50,edgecolors="k")
plotBoundary(X,clf)

# 境界線を引く関数の定義

def plotBoundary2(X, clf, boundary=True):
    colors = ['k'];
    linestyles = ['-'];
    levels = [0];

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]) # evaluate the value

    n_classes = Z.shape[1]

    for c in range(n_classes):
        Zc = Z[:,c].reshape(XX.shape) # just reshape

        if boundary:
            plt.contour(XX, YY, Zc,
                        colors=colors,
                        linestyles=linestyles,
                        levels=levels) # draw lines (level=0:boundary)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(X,y)

plt.scatter(X[:,0],X[:,1],marker='o',c=y,s=50,edgecolors="k")
plotBoundary(X,clf)
plt.show()

plt.scatter(X[:,0],X[:,1],marker='o',c=y,s=50,edgecolors="k")
plotBoundary2(X,clf)
clf


# 境界線を引く関数の定義

def plotBoundary3(X, clf, mesh=True, boundary=True):
    colors = ['k'];
    linestyles = ['-.', '-', '--'];
    levels = [-1,0,1];
    cmaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples]

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]) # evaluate the value
    print(Z.shape)
    n_classes = Z.shape[1]

    for c in range(n_classes):
        Zc = Z[:,c].reshape(XX.shape) # just reshape
        plt.show()
        if mesh:
            plt.pcolormesh(XX, YY, Zc, zorder=-10, cmap=cmaps[c])
            plt.colorbar()

        if boundary:
            plt.contour(XX, YY, Zc,
                        colors=colors,
                        linestyles=linestyles,
                        levels=levels)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
plotBoundary3(X, clf)

# 境界線を引く関数の定義

#MaskedArray
import numpy.ma as ma

def plotBoundary4(X, clf, mesh=True, boundary=True):
    colors = ['k'];
    linestyles = ['-.', '-', '--'];
    levels = [-1,0,1];
    cmaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples]

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]) # evaluate the value

    n_classes = Z.shape[1]

    Zmax = Z.argmax(axis=1).reshape(XX.shape)

    for c in range(n_classes):
        Zc = ma.array(Z[:,c].reshape(XX.shape), mask=(Zmax != c))

        if mesh:
            plt.pcolormesh(XX, YY, Zc, zorder=-10, cmap=cmaps[c])

        if boundary:
            plt.contour(XX, YY, Zc,
                        colors=colors,
                        linestyles=linestyles,
                        levels=levels)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

plt.scatter(X[:,0],X[:,1],marker="o",s=50,c=y,edgecolors="k")
plotBoundary4(X,clf)

clf = svm.SVC(kernel="linear",
        decision_function_shape='ovr',C=10)
clf.fit(X,y)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
plotBoundary3(X, clf)
plt.show()

plotBoundary4(X, clf)

clf  = svm.SVC(kernel='rbf',
               decision_function_shape='ovr',C=10)
clf.fit(X,y)
plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
plotBoundary3(X, clf)
plt.show()
plotBoundary4(X, clf)

#ovo or ovr

# 4クラスのデータセットを準備
X, y = make_classification(n_samples=20, # 20個生成
                           n_features=2, # 2次元
                           n_classes=4, # 4クラス
                           n_clusters_per_class=1,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           random_state=8   # 乱数種（再現用）
                           )

from matplotlib.colors import ListedColormap as lcmap
brgp = lcmap(['blue','red','green','purple'])

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=brgp, edgecolors='k');

clf = svm.SVC(kernel='linear',
              decision_function_shape='ovr') # one-vs-rest, one-vs-all
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, cmap=brgp, edgecolors='k')
plotBoundary(X, clf, cmap=brgp)
plt.show()

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, cmap=brgp, edgecolors='k')
plotBoundary3(X, clf)

clf = svm.SVC(kernel='linear',
              decision_function_shape='ovo') # one-vs-one
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, cmap=brgp, edgecolors='k')
plotBoundary(X, clf, cmap=brgp)
plt.show()

plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, cmap=brgp, edgecolors='k')
plotBoundary3(X, clf, mesh=False)

from itertools import combinations

list(combinations([0,1,2,3],2))

list(combinations("ABCD",2))

list(combinations("ABCDEFG",2))
