import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import fetch_20newsgroups

newsgroup_train = fetch_20newsgroups(subset='train')
newsgroup_test = fetch_20newsgroups(subset='test')

newsgroup_train.target_names
newsgroup_train.target.size
newsgroup_test.target.size
print(newsgroup_train.data[3])
# テキストから特徴量ベクトルに変換する
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(newsgroup_train.data)
X_train.shape
X_test = vectorizer.transform(newsgroup_test.data)
X_test.shape

y_train = newsgroup_train.target
y_test = newsgroup_test.target

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

X_train[0]
X_train_0 = X_train[0].toarray()
X_train_0

np.count_nonzero(X_train_0)
X_train_0.shape[1]
X_train_0[np.nonzero(X_train_0)]
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(np.sort(X_train_0[np.nonzero(X_train_0)]))

from sklearn.pipeline import Pipeline
# 特徴量を2値化する処理
from sklearn.preprocessing import Binarizer

pipe = Pipeline([("bin",Binarizer()),
                 ("clf",LogisticRegression())])
from sklearn.model_selection import GridSearchCV

param = {"bin__threshold":[0.001,0.01,0.05,0.1,0.2,0.3,0.4]}

gs1 = GridSearchCV(pipe,param,n_jobs=-1,verbose=2)

gs1.fit(X_train,y_train)
gs1.best_params_, gs1.best_score_, gs1.best_estimator_
gs1.score(X_test,y_test)

plt.errorbar(gs1.cv_results_['param_bin__threshold'].data,
             gs1.cv_results_['mean_train_score'],
             yerr=gs1.cv_results_['std_train_score'],
             label="training")

plt.errorbar(gs1.cv_results_['param_bin__threshold'].data,
             gs1.cv_results_['mean_test_score'],
             yerr=gs1.cv_results_['std_test_score'],
             label="test(val)")

plt.ylim(0, 1.01)
plt.xlabel("threshold")
plt.ylabel("accuracy")
plt.legend(loc="best");

pipe = Pipeline([('bin', Binarizer()),
                 ('clf', LogisticRegression())])

param = {'bin__threshold': [0.001, 0.01, 0.05],
         'clf__C': 10**np.arange(1.0, 10.0) }

from sklearn.model_selection import RandomizedSearchCV

gs11 = RandomizedSearchCV(pipe, param, n_jobs=-1, verbose=2)
gs11.fit(X_train, y_train)
gs11.best_params_, gs11.best_score_, gs11.best_estimator_
gs11.score(X_test, y_test)


from sklearn.svm import LinearSVC

pipe = Pipeline([("bin",Binarizer()),
                ("clf",LinearSVC())])

param = {"bin__threshold":[0.001,0.01,0.05],
         "clf__C":10**np.arange(1.0,10.0)}

from sklearn.model_selection import RandomizedSearchCV

gs2 = RandomizedSearchCV(pipe,param,n_jobs=-1,verbose=2)
gs2.fit(X_train,y_train)
gs2.best_params_,gs2.best_score_,gs2.best_estimator_
gs2.score(X_test,y_test)


from sklearn.linear_model import SGDClassifier

pipe = Pipeline([('bin', Binarizer()),
                 ('clf', SGDClassifier(loss="hinge") )])

param = {'bin__threshold': [0.001, 0.01, 0.05],
         'clf__alpha': 10**np.arange(-10.0, -1.0) }

from sklearn.model_selection import RandomizedSearchCV

gs22 = RandomizedSearchCV(pipe, param, n_jobs=-1, verbose=2)
gs22.fit(X_train, y_train)
gs22.best_params_,gs22.best_score_,gs22.best_estimator_
gs22.score(X_test,y_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_pred = gs22.predict(X_test)

print(classification_report(y_test,y_pred,digits=4))

conf_mat = confusion_matrix(y_test,y_pred)
conf_mat

import matplotlib.pyplot as plt
%matplotlib inline
plt.gray()

plt.imshow(1- conf_mat / conf_mat.sum(axis=1),
           interpolation='nearest')

plt.yticks(range(20), newsgroup_train.target_names);
plt.xticks(range(20), newsgroup_train.target_names, rotation=90);
