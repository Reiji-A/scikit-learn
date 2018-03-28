import warnings
warnings.filterwarnings("ignore")
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf
# 正則化:1.0
clf.C = 1.0
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# 正則化:100.0
clf.C = 100.0
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# 正則化を変化させて汎化性能を調査
C_range = [1e-5,1e-3,1e-2,1,1e2,1e5,1e10]

C_range_exp = np.arange(-15.0,21.0)
C_range = 10 ** C_range_exp
C_range.shape
C_range

from sklearn.model_selection import GridSearchCV

param = {"C":C_range}#clf.C

gs = GridSearchCV(clf,param)
gs.fit(X_train,y_train)
gs.cv_results_
gs.best_params_,gs.best_score_,gs.best_estimator_
# gs.best_score_:three holdでバリデーション後のスコアを表示

clf_best = gs.best_estimator_
clf_best.score(X_test,y_test)
# 上記のように代入しないで下記のようにスコアを表示可能
gs.score(X_test,y_test)

plt.errorbar(gs.cv_results_["param_C"].data,
         gs.cv_results_["mean_train_score"],
         yerr = gs.cv_results_["std_train_score"],
         label = "training")

plt.errorbar(gs.cv_results_["param_C"].data,
         gs.cv_results_["mean_test_score"],
         yerr = gs.cv_results_["std_test_score"],
         label = "test(val)")

plt.ylim(.6,1.01)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend(loc="best")

plt.errorbar(gs.cv_results_['param_C'].data,
             gs.cv_results_['mean_fit_time'],
             yerr=gs.cv_results_['std_fit_time'],
             label="training")

plt.errorbar(gs.cv_results_['param_C'].data,
             gs.cv_results_['mean_score_time'],
             yerr=gs.cv_results_['std_score_time'],
             label="test(val)")

plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("computation time")
plt.legend(loc="best");

from sklearn.svm import SVC

clf  = SVC()
clf
C_range_exp = np.arange(-2.0,5.0)
C_range = 10 ** C_range_exp
C_range
param = {"C":C_range,
        "kernel":["linear","rbf"]}

gs = GridSearchCV(clf,param,n_jobs=-1,verbose=2)
gs.fit(X_train,y_train)
gs.best_params_,gs.best_score_,gs.best_estimator_
gs.cv_results_
s_linear = [gs.cv_results_["param_kernel"]=="linear"]

plt.plot(gs.cv_results_["param_C"][s_linear].data,
         gs.cv_results_["mean_train_score"][s_linear],
         label = "training(linear)")

plt.plot(gs.cv_results_["param_C"][s_linear].data,
         gs.cv_results_["mean_test_score"][s_linear],
         linestyle = "--",
         label = "test/val(linear)")

s_rbf = [gs.cv_results_["param_kernel"]=="rbf"]

plt.plot(gs.cv_results_['param_C'][s_rbf].data,
         gs.cv_results_['mean_train_score'][s_rbf],
         label="training (rbf)")

plt.plot(gs.cv_results_['param_C'][s_rbf].data,
         gs.cv_results_['mean_test_score'][s_rbf],
         linestyle="--",
         label="test/val (rbf)")

plt.ylim(.6, 1.01)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
gs.score(X_test, y_test)
SVC(kernel='rbf').gamma


from sklearn.svm import SVC

clf = SVC()
clf
C_range_exp = np.arange(-2.0, 10.0)
C_range = 10 ** C_range_exp

gamma_range_exp = np.arange(-10.0, 0.0, 3)
gamma_range = 10 ** gamma_range_exp

param = [ {'C': C_range,
           'kernel': ['linear']},

          {'C': C_range,
           'gamma': gamma_range,
           'kernel': ['rbf']} ]

gs = GridSearchCV(clf, param, n_jobs=-1, verbose=2)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_, gs.best_estimator_

s_linear = [gs.cv_results_['param_kernel']=='linear']

plt.plot(gs.cv_results_['param_C'][s_linear].data,
         gs.cv_results_['mean_train_score'][s_linear],
         label="training (linear)")

plt.plot(gs.cv_results_['param_C'][s_linear].data,
         gs.cv_results_['mean_test_score'][s_linear],
         linestyle="--",
         label="test/val (linearr)")

s_rbf = [gs.cv_results_['param_kernel']=='rbf']

for g in gamma_range:
    s_gamma = gs.cv_results_['param_gamma'][s_rbf].data == g

    plt.plot(gs.cv_results_['param_C'][s_rbf][s_gamma].data,
             gs.cv_results_['mean_train_score'][s_rbf][s_gamma],
             label="training (rbf, gamma {0:.0e})".format(g))

    plt.plot(gs.cv_results_['param_C'][s_rbf][s_gamma].data,
             gs.cv_results_['mean_test_score'][s_rbf][s_gamma],
             linestyle="--",
             label="test/val (rbf, gamma {0:.0e})".format(g))

plt.ylim(.6, 1.01)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
gs.score(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf
param = {'n_neighbors': range(1,20) }

gs = GridSearchCV(clf, param)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_, gs.best_estimator_

plt.errorbar(gs.cv_results_['param_n_neighbors'].data,
             gs.cv_results_['mean_train_score'],
             yerr=gs.cv_results_['std_train_score'],
             label="training")

plt.errorbar(gs.cv_results_['param_n_neighbors'].data,
             gs.cv_results_['mean_test_score'],
             yerr=gs.cv_results_['std_test_score'],
             label="test(val)")

plt.ylim(.6, 1.01)
plt.xlabel("# neighbors")
plt.ylabel("accuracy")
plt.legend(loc="best");
gs.score(X_test, y_test)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=2000)
clf

param = {'hidden_layer_sizes': [(10,), (50,), (100,),
                                (10,10,), (50,50,), (100,100,),
                                (10, 5,), (5,5,), (30, 20, 10),
                                (100,1000,50,), (1000,100,50,),
                                (10,10,10), (50,50,50), (100,100,100,),
                                ],
          'activation' : ['identity', 'logistic', 'tanh', 'relu'],
          'beta_1' : [0.9, 0.8, 0.7, 0.6, 0.5],
          'beta_2' : [0.999, 0.9, 0.8, 0.7],
          'alpha' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        }
gs = GridSearchCV(clf, param, n_jobs=-1, verbose=1)
# gs.fit(X_trian,y_train)
# 6720通りのため終わらない
# 終わらないため、ランダムにパラメータを取ってくる下記オブジェクトを利用してみる
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(clf, param,
                        n_iter=20,
                        n_jobs=-1, verbose=2)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_, gs.best_estimator_
gs.score(X_test, y_test)
