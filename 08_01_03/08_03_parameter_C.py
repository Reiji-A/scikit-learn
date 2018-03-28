import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.set_cmap(plt.cm.Paired)

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
df = pd.DataFrame(X_train,columns=data.feature_names)
df.ix[:,0:2].describe()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
df_scaled = pd.DataFrame(X_train,columns=data.feature_names)
df_scaled.ix[:,0:2].describe()

from sklearn.svm import SVC
clf = SVC(kernel="rbf")
clf
clf.C = 1
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
clf.C = 1000
clf.fit(X_test,y_test)
clf.score(X_test,y_test)
clf.predict(X_test)
y_test

import time

C_range_exp = np.arange(-20,20.)
C_range = 10 ** C_range_exp

scores = []
comp_time = []

for C in C_range:
    clf.C = C

    st = time.time()
    clf.fit(X_train,y_train)
    comp_time.append(time.time() - st)

    score = clf.score(X_test,y_test)
    print(C,score)
    scores.append(score)

scores = np.array(scores)
scores
comp_time = np.array(comp_time)
comp_time

plt.plot(C_range_exp,scores)
plt.ylabel("accuracy")
plt.xlabel(r"$\log_{10}$(C)")
plt.show()

plt.plot(C_range_exp,comp_time)
plt.ylabel("computation time[sec]")
plt.xlabel(r"$\log_{10}$C")
plt.show()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.C = 1
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

clf.C = 100
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

C_range_exp = np.arange(-20,15.)
C_range = 10 ** C_range_exp

scores = []
comp_time = []

for C in C_range:
    clf.C = C

    st = time.time()
    clf.fit(X_train, y_train)
    comp_time.append(time.time() - st)

    score = clf.score(X_test, y_test)
    print(C, score)
    scores.append(score)

scores = np.array(scores)
comp_time = np.array(comp_time)

plt.plot(C_range_exp, scores)
plt.ylabel("accuracy")
plt.xlabel(r"$\log_{10}$(C)");
plt.show()

plt.plot(C_range_exp, comp_time)
plt.ylabel("computation time [sec]")
plt.xlabel(r"$\log_{10}$(C)");


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, # 20個生成
                  n_features=2, # 2次元
                  centers=2,    # クラスタ中心2個
                  cluster_std =2, # クラスタの大きさ（標準偏差）
                  random_state=3   # 乱数種（再現用）
                 )
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k'); # 2次元散布図でプロット


def plotSVMBoundary(X, clf, mesh=True, boundary=True):

    # if SVM, draw margine lines
    colors = ['k']*3
    linestyles = ['-']*3
    levels = [-1, 0, 1]
    # if SVM, plot support vecters
    plt.scatter(clf.support_vectors_[:, 0],
                clf.support_vectors_[:, 1],
                s=80, facecolors='none', edgecolors='k')

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]) # evaluate the value
    Z = Z.reshape(XX.shape) # just reshape

    if mesh:
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired, zorder=-10)
    if boundary:
        plt.contour(XX, YY, Z,
                    colors=colors,
                    linestyles=linestyles,
                    levels=levels) # draw lines (level=0:boundary, level=+-1:margine lines)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

from sklearn.svm import SVC

clf = SVC()
clf.kernel = 'rbf'

for C in [1e-10, 1e-5, 1, 1e5, 1e10, 1e20]:

    clf.C = C
    clf.fit(X,y)
    plotSVMBoundary(X, clf, mesh=True)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
    plt.title("C={0:.0e}".format(C))

    plt.show()

from sklearn.svm import SVC

clf = SVC()
clf.kernel = 'rbf'

for C in 10 ** np.arange(-1.0, 6.0):

    clf.C = C
    clf.fit(X,y)
    plotSVMBoundary(X, clf, mesh=True)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=50, c=y, edgecolors='k')
    plt.title("C={0:.0e}".format(C))

    plt.show()

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, # 20個生成
                  n_features=2, # 2次元
                  centers=[(-2, -2), (2, 2)],    # クラスタ中心2個
                  cluster_std=2, # クラスタの大きさ（標準偏差）
                  # random_state=3   # 乱数種（再現用）
                 )
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k'); # 2次元散布図でプロット
plt.xlim(-10,10)
plt.ylim(-10,10)

from sklearn.svm import SVC

clf = SVC()
clf.kernel = 'rbf'

X_train, y_train = make_blobs(n_samples=200, # 20個生成
              n_features=2, # 2次元
              centers=[(-2, -2), (2, 2)],    # クラスタ中心2個
              cluster_std=2, # クラスタの大きさ（標準偏差）
              # random_state=3   # 乱数種（再現用）
             )

X_test, y_test = make_blobs(n_samples=200, # 20個生成
      n_features=2, # 2次元
      centers=[(-2, -2), (2, 2)],    # クラスタ中心2個
      cluster_std=2, # クラスタの大きさ（標準偏差）
      # random_state=3   # 乱数種（再現用）
     )


for C in 10 ** np.arange(-1.0, 6.0):

    clf.C = C

    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=50, c=y, label="training", edgecolors='k')

    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)

    plotSVMBoundary(X_train, clf, mesh=True)

    plt.scatter(X_test[:, 0], X_test[:, 1], marker='*', s=50, c=y, label="test", edgecolors='k')

    test_score = clf.score(X_test, y_test)

    plt.title("C={0:.0e}, train {1}, test {2}".format(C, train_score, test_score))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
