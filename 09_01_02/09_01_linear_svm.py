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

import time

from sklearn.svm import SVC
clf = SVC(kernel='linear')

clf.C = 1
st = time.time()
clf.fit(X_train,y_train)
print(1000*(time.time()-st),"[ms]")

clf.score(X_test,y_test)

clf.C = 1e20
st = time.time()
clf.fit(X_train,y_train)
print(1000*(time.time()-st),"[ms]")

clf.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV

param = {"C":10**np.arange(-15.0,21.0)}

gs = GridSearchCV(clf,param,verbose=1)
gs.fit(X_train,y_train)

gs.score(X_test,y_test)
gs.best_params_,gs.best_estimator_

plt.plot(gs.cv_results_['param_C'].data,
         gs.cv_results_['mean_fit_time'],
         label="training")

plt.plot(gs.cv_results_['param_C'].data,
         gs.cv_results_['mean_score_time'],
         label="test(val)")

plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("computation time [sec]")
plt.legend(loc="best");
plt.title("SVM with linear kernel");

from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.C = 1

st = time.time()
clf.fit(X_train,y_train)
print(1000*(time.time()-st),"[ms]")
clf.score(X_test,y_test)

clf.C = 1e20

st = time.time()
clf.fit(X_train,y_train)
print(1000*(time.time()-st),"[ms]")

clf.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV

param = {"C":10**np.arange(-15.0,21.0)}

gs2 = GridSearchCV(clf,param,verbose=1)
gs2.fit(X_train,y_train)
gs2.best_params_,gs2.best_score_,gs2.best_estimator_
gs2.cv_results_

plt.plot(gs2.cv_results_['param_C'].data,
         gs2.cv_results_['mean_fit_time'],
         label="training")

plt.plot(gs2.cv_results_['param_C'].data,
         gs2.cv_results_['mean_score_time'],
         label="test(val)")

plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("computation time [sec]")
plt.legend(loc="best")
plt.title("LinearSVM");

plt.plot(gs.cv_results_['param_C'].data,
         gs.cv_results_['mean_fit_time'],
         label="training SVM (linear kernel)")

plt.plot(gs2.cv_results_['param_C'].data,
         gs2.cv_results_['mean_fit_time'],
         label="training LinearSVM")

plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("computation time [sec]")
plt.legend(loc="best");


from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

estimators = [('scaler', MinMaxScaler()),
              ('clf', LinearSVC())]

pipe = Pipeline(estimators)

from sklearn.model_selection import GridSearchCV

param = {"clf__C":10**np.arange(-15.0,21.0)}

gs = GridSearchCV(pipe,param,verbose=1)
gs.fit(X_train,y_train)
gs.best_params_,gs.best_score_,gs.best_estimator_
gs.cv_results_
gs.score(X_test,y_test)

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_fit_time'],
         label="training")

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_score_time'],
         label="test(val)")
plt.ylabel("computation time [sec]")
plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt.legend(loc="upper left");

plt.twinx()

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_train_score'],
         linestyle="--",
         label="training")

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_test_score'],
         linestyle="--",
         label="test(val)")
plt.ylabel("accuracy")
plt.legend(loc="lower right");

plt.title("LinearSVM")

from sklearn.preprocessing import MinMaxScaler

estimators = [('scaler', MinMaxScaler()),
              ('clf', SVC(kernel='linear'))]

pipe = Pipeline(estimators)

from sklearn.model_selection import GridSearchCV

param = {'clf__C': 10**np.arange(-15.0,21.0)}

gs = GridSearchCV(pipe, param, verbose=1)
gs.fit(X_train, y_train)
gs.best_params_,gs.best_score_,gs.best_estimator_
gs.cv_results_

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_fit_time'],
         label="training")

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_score_time'],
         label="test(val)")
plt.ylabel("computation time [sec]")
plt.ylim(0,)
plt.xscale("log")
plt.xlabel("C")
plt.legend(loc="upper left");

plt.twinx()

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_train_score'],
         linestyle="--",
         label="training")

plt.plot(gs.cv_results_['param_clf__C'].data,
         gs.cv_results_['mean_test_score'],
         linestyle="--",
         label="test(val)")
plt.ylabel("accuracy")
plt.legend(loc="lower right");

plt.title("SVM with linear kernel");

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
mnist.data.shape
# MNISTの場合，60000が学習，10000がテスト，と決まっている
# http://yann.lecun.com/exdb/mnist/

X_train,X_test = mnist.data[:60000]/255,mnist.data[60000:]/255
y_train,y_test = mnist.target[:60000],mnist.target[60000:]
clf = SVC(kernel="linear")

st = time.time()

for thin in [100,50,10,5,4]:#>1 min...
    st = time.time()
    clf.fit(X_train[::thin],y_train[::thin])
    et = time.time() -st
    score = clf.score(X_test[::thin],y_test[::thin])
    print("{:.2f}sec,size{},accuracy{}".format(et,
         y_train[::thin].size,
         score))

clf = LinearSVC(dual=False)
for thin in [100,50,10,5,4]:#>1 min...
    st = time.time()
    clf.fit(X_train[::thin],y_train[::thin])
    et = time.time() -st
    score = clf.score(X_test[::thin],y_test[::thin])
    print("{:.2f}sec,size{},accuracy{}".format(et,
         y_train[::thin].size,
         score))
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import time
import matplotlib.pyplot as plt
for clf in [SVC(kernel='linear'),
            LinearSVC(),
            LinearSVC(dual=False)]:
    times = []
    sizes = []
    for thin in [100,50,10,5,4]:
        st = time.time()
        clf.fit(X_train[::thin],y_train[::thin])
        times.append(time.time() - st)
        sizes.append(y_train[::thin].size)
    plt.plot(sizes,times,label=clf.__class__.__name__)
plt.legend(loc="best")
plt.show()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
estimators = [("scaler",MinMaxScaler()),
               ("clf",LinearSVC(dual=False))]
pipe = Pipeline(estimators)

param = {"clf__C":10**np.arange(-5,0,10.0)}
import numpy as np
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(pipe,param,n_jobs=-1,verbose=2)
# gs.fit(X_train,y_train)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
estimators = [("scaler",MinMaxScaler()),
               ("clf",LinearSVC(dual=True))]
pipe = Pipeline(estimators)

param = {"clf__C":10**np.arange(-5,0,10.0)}
import numpy as np
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(pipe,param,n_jobs=-1,verbose=2)
# gs.fit(X_train,y_train)

estimators = [('scaler', MinMaxScaler()),
              ('clf', SVC(kernel='linear'))]

pipe = Pipeline(estimators)

param = {'clf__C': 10**np.arange(-5.0,10.0)}

gs = GridSearchCV(pipe, param, n_jobs=-1, verbose=2)
# gs.fit(X_train, y_train) # 2 hours ?
