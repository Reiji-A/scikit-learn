import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
import pandas as pd

df = pd.DataFrame(data.data,data.target,columns=data.feature_names)
df
print(data.DESCR)

import matplotlib.pyplot as plt
%matplotlib inline

data.feature_names[4]
data.feature_names[3]
df["mean smoothness"].describe()
df["mean area"].describe()
plt.scatter(data.data[:,3],data.data[:,4],c=data.target)
plt.xlabel(data.feature_names[3])
plt.ylabel(data.feature_names[4])

plt.scatter(data.data[:,3],data.data[:,4],c = data.target)
plt.xlim(0,3000)
plt.ylim(0,3000)
plt.xlabel(data.feature_names[3])
plt.ylabel(data.feature_names[4])

# data全体で実行
X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits = 1,
                  train_size = 0.8,
                  test_size = 0.2,
                  random_state=0)
train_index,test_index = next(ss.split(X,y))
X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_train_scale.mean(axis=0)
X_train_scale.std(axis=0)

X_test_scale = scaler.transform(X_test)
X_test_scale.mean(axis=0),X_test_scale.std(axis=0)

plt.scatter(X_train_scale[:,3],X_train_scale[:,4],c="blue",label="train")
plt.scatter(X_test_scale[:,3],X_test_scale[:,4],c="red",label="test")
plt.xlabel(data.feature_names[3]+"(standardised)")
plt.ylabel(data.feature_names[4]+"(standardised)")
plt.legend(loc="best")
from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(X_train_scale,y_train)
clf.score(X_test_scale,y_test)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# range scaling
from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler([-1,1])
mmscaler.fit(X_train)
X_train_mms = mmscaler.transform(X_train)
X_test_mms = mmscaler.transform(X_test)
X_train_mms.max(axis = 0),X_train_mms.min(axis = 0)
X_train.max(axis = 0),X_test.min(axis = 0)
plt.scatter(X_train_mms[:,3],X_train_mms[:,4],c = "blue",label="train")
plt.scatter(X_test_mms[:,3],X_test_mms[:,4],c = "red",label="test")
plt.xlabel(data.feature_names[3]+"(scaled)")
plt.ylabel(data.feature_names[4]+"(scaled)")
plt.legend(loc="best")

clf.fit(X_train_mms,y_train)
clf.score(X_test_mms,y_test)

# Normalization
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

normalizer.fit(X_train)

X_train_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)

X_train_norm.max(),X_train_norm.min()
X_test_norm.max(),X_test_norm.min()

np.linalg.norm(X_train,axis=1)[:20]

np.linalg.norm(X_train_norm,axis=1)[:20]
clf.fit(X_train_norm,y_train)
clf.score(X_test_norm,y_test)

plt.scatter(X_train_norm[:, 3],
            X_train_norm[:, 4], c='blue',
            label="train")
plt.scatter(X_test_norm[:, 3],
            X_test_norm[:, 4],  c='red',
            label="test")
plt.xlabel(data.feature_names[3] + " (normalized)")
plt.ylabel(data.feature_names[4] + " (normalized)")
plt.legend(loc="best");

for norm in ["l2","l1","max"]:
    normalizer = Normalizer(norm=norm)
    normalizer.fit(X_train)
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    clf.fit(X_train_norm,y_train)
    print(norm,clf.score(X_test_norm,y_test))

# PCA Whitening
data.feature_names[6],data.feature_names[7]
plt.scatter(data.data[:,6],data.data[:,7])
X = data.data[:,[6,7]]
y = data.target
X[:,0].max()
X[:,1].max()
plt.scatter(X[:,0],X[:,1])
plt.xlim(0,0.5)
plt.ylim(0,0.5)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:,0],X_new[:,1])
plt.xlim(-0.1,0.4)
plt.ylim(-0.25,0.25)

pca = PCA(whiten=True)
pca.fit(X)

X_new = pca.transform(X)
plt.scatter(X_new[:,0],X_new[:,1])
plt.xlim(-4,10)
plt.ylim(-4,10)
X_new.mean(axis = 0),X_new.std(axis = 0)

# ZCA Whitening
X = np.random.uniform(low = -1,high = 1,size=(1000,2))*(2,1)
X.shape
y = 2 * X[:,0] + X[:,1]
y.shape
plt.scatter(X[:,0],X[:,1],c = y)
plt.xlim(-3,3)
plt.ylim(-3,3)
angle = np.pi/4
R = np.array([[np.sin(angle),-np.cos(angle)],[np.cos(angle),np.sin(angle)]])
R

X_rot  = X.dot(R)

plt.scatter(X_rot[:,0],X_rot[:,1],c = y)
plt.xlim(-3,3)
plt.ylim(-3,3)

X = X_rot

pca = PCA(whiten=False)
pca.fit(X)
X_new = pca.transform(X)
plt.scatter(X_new[:,0],X_new[:,1],c = y)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title("PCA Whitening")

X_new2 = X_new.dot(pca.components_)
plt.scatter(X_new2[:,0],X_new2[:,1],c = y)
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title("ZCA Whitening")
pca.components_
