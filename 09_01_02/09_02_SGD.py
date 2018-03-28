import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import time

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")

# MNISTの場合、60000が学習、10000がテスト

X_train,X_test = mnist.data[:60000]/255,mnist.data[60000:]/255
y_train,y_test = mnist.target[:60000],mnist.target[60000:]

X_train.shape
X_test.shape

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="log") #linear SVM
#損失関数:log->logistic回帰
#100,50....個ずつ間引いで学習
for thin in [100,50,10,5,4,3,2,1]:
    st = time.time()
    clf.fit(X_train[::thin],y_train[::thin])
    et = time.time() - st
    score = clf.score(X_test[::thin],y_test[::thin])
    print("{0:.2f}sec,size{1},accuracy{2}".format(et,
                                               y_train[::thin].size,
                                               score))

clf = SGDClassifier(loss="hinge") #linear SVM

for thin in [100,50,10,5,4,3,2,1]:
    st = time.time()
    clf.fit(X_train[::thin],y_train[::thin])
    et = time.time() - st
    score = clf.score(X_test[::thin],y_test[::thin])
    print("{0:.2f}sec,size{1},accuracy{2}".format(et,
                                               y_train[::thin].size,
                                               score))

from sklearn.svm import LinearSVC

for clf in[SGDClassifier(loss="hinge"),
          LinearSVC(dual=False)]:
    times = []
    sizes = []
    for thin in [100,50,10,5,3,2,1]:
        st = time.time()
        clf.fit(X_train[::thin],y_train[::thin])
        times.append(time.time()-st)
        sizes.append(y_train[::thin].size)
    plt.plot(sizes,times,label=clf.__class__.__name__)
plt.legend(loc="best")
plt.ylabel("computation time [sec]")
plt.xlabel("# samples")
plt.show()


C_range_exp = np.arange(-5.0,15.0)
C_range = 10 ** C_range_exp

scores = []
comp_time = []

clf = SGDClassifier(loss = "hinge") #SVM

for C in C_range:
    clf.alpha = X_train.shape[0] / C # n_sample/alpha = C http://scikit-learn.org/stable/modules/svm.html#svc

    st = time.time()
    clf.fit(X_train, y_train)
    et = time.time() - st

    comp_time.append(et)
    score = clf.score(X_test, y_test)
    scores.append(score)

    print(C, et, score)

scores = np.array(scores)
comp_time = np.array(comp_time)
scores
comp_time

plt.plot(C_range_exp,scores)
plt.ylabel("accuracy")
plt.xlabel("C")
plt.ylim(0,1)

plt.show()

plt.plot(C_range_exp,comp_time)
plt.ylim(0,)
plt.ylabel("computation time [sec]")
plt.xlabel("C")

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

estimators = [("scaler",MinMaxScaler()),
              ("clf",SGDClassifier(loss="hinge"))]
pipe = Pipeline(estimators)

param = {'clf__alpha': (X_train.shape[0] * 2./3.) / (10**np.arange(-5.0, 10.0)) }

gs = GridSearchCV(pipe, param, n_jobs=-1, verbose=2)
gs.fit(X_train, y_train)

plt.plot(gs.cv_results_['param_clf__alpha'].data,
         gs.cv_results_['mean_fit_time'],
         label="training")

plt.plot(gs.cv_results_['param_clf__alpha'].data,
         gs.cv_results_['mean_score_time'],
         label="test(val)")
plt.ylabel("computation time [sec]")
plt.ylim(0,)
plt.xscale("log")
plt.xlabel("alpha = #sample / C")
plt.legend(loc="upper left");

plt.twinx()

plt.plot(gs.cv_results_['param_clf__alpha'].data,
         gs.cv_results_['mean_train_score'],
         linestyle="--",
         label="training")

plt.plot(gs.cv_results_['param_clf__alpha'].data,
         gs.cv_results_['mean_test_score'],
         linestyle="--",
         label="test(val)")
plt.ylabel("accuracy")
plt.legend(loc="lower right");

plt.title("SGDClassifier")



from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size=0.2,
                  random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge")

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

clf = SGDClassifier(loss="log")

clf.fit(X_train,y_train)
clf.score(X_test,y_test)
