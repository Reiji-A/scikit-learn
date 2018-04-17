import numpy as np
datapath = "/Users/macuser/Documents/Python/scikit-learn/udemy_machine_learning/20170628ipynb/"
with open(datapath + "2D_example.csv") as f:
    print(f.read())

data = np.loadtxt(datapath + "2D_example.csv",delimiter=",")
# 20個の特徴量,1列目が目的変数、2列目,3列目が説明変数
data.shape
data[:,0]# 目的変数
data[:,1:3]# 説明変数
y = data[:,0].astype(int)# 1列目の目的変数をを整数(int)に変換
X = data[:,1:3]#2,3列目の説明変数を取り出し
y

import matplotlib.pyplot as plt
%matplotlib inline
plt.set_cmap(plt.cm.Paired) #色設定
X[:,0]
X[:,1]

# 説明変数を目的変数で分類し描画
plt.scatter(X[:,0],X[:,1],c=y,s=50)

#colorはyの0or1,sizeは50
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

# k近傍法で学習
from sklearn import neighbors
# データポイントを1対1で比較するパラメータ:n_neighbors=1
clf = neighbors.KNeighborsClassifier(n_neighbors=1).fit(X,y)
clf
plt.scatter(X[:,0],X[:,1],marker="o",c=y,s=50,edgecolors="k")
plotBoundary(X,clf)
# ロジスティック回帰(線形回帰)で分類
from sklearn import linear_model

clf = linear_model.LogisticRegression().fit(X,y)
clf
plt.scatter(X[:,0],X[:,1],marker="o",c=y,s=50,edgecolors="k")
plotBoundary(X,clf)
# サポートベクトマシーン(線形)で分類
from sklearn import svm
clf = svm.SVC(kernel='linear').fit(X,y)
clf
plt.scatter(X[:,0],X[:,1],marker="o",c=y,s=50,edgecolors="k")
plotBoundary(X,clf)
# サポートベクトマシン(Gaussianカーネル)で分類
clf = svm.SVC(kernel='rbf').fit(X,y)
clf
plt.scatter(X[:,0],X[:,1],marker="o",c=y,s=50,edgecolors="k")
plotBoundary(X,clf)
