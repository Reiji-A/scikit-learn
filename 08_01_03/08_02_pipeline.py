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
X_train.shape
from sklearn.decomposition import PCA

pca = PCA(whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

pca_2 = PCA(whiten=False)
pca_2.fit(X_train)
X_train_pca_2 = pca_2.transform(X_train)
X_test_pca_2 = pca_2.transform(X_test)
import pandas as pd
df_2 = pd.DataFrame(X_train_pca_2)
df_2.ix[:,0:1].describe()
plt.scatter(df_2[0],df_2[1])

df = pd.DataFrame(X_train_pca)
df.ix[:,0:1].describe()
plt.scatter(df[0],df[1])
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

clf.fit(X_train_pca,y_train)
clf.score(X_test_pca,y_test)

from sklearn.pipeline import Pipeline

estimators = [("pca",PCA(whiten=True)),
              ("clf",LogisticRegression())]
pipe = Pipeline(estimators)

pipe.fit(X_train,y_train)
pipe.score(X_test,y_test)


from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

estimators = [("mms",MinMaxScaler()),
             ("clf",SVC(kernel='rbf',C=1e10))]
pipe = Pipeline(estimators)

pipe.fit(X_train,y_train)

pipe.score(X_test,y_test)

estimators = [("pca",PCA(whiten=True)),
             ("clf",LogisticRegression())]
pipe = Pipeline(estimators)

from sklearn.model_selection import GridSearchCV

param = {'clf__C':[1e-5, 1e-3, 1e-2, 1, 1e2, 1e5, 1e10]} # clf.C

gs = GridSearchCV(pipe, param)
gs.fit(X_train, y_train)

gs.best_params_,gs.best_score_,gs.best_estimator_
gs.score(X_test,y_test)


from sklearn.svm import SVC

C_range = [1e-3,1e-2,1,1e2,1e3]

param = {'clf__C': C_range,
         'clf__kernel': ['linear', 'rbf'],
         'pca__whiten': [True, False],
         'pca__n_components': [30, 20, 10]}

estimators = [('pca', PCA()),
              ('clf', SVC())]

pipe = Pipeline(estimators)

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(pipe,param,n_jobs=-1,verbose=2)

gs.fit(X_train,y_train)
gs.best_params_,gs.best_score_,gs.best_estimator_
gs.score(X_test,y_test)
