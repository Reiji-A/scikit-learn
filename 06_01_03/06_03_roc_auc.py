import numpy as np
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

X_train.shape

from sklearn import linear_model
clf = linear_model.LogisticRegression()

clf.fit(X_train, y_train)

clf.score(X_test,y_test)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_test, y_pred)

cmat = confusion_matrix(y_test, y_pred)
cmat
X_test[12:15],y_test[12:15]
clf.decision_function(X_test[2:5])
clf.predict(X_test[2:5])

(clf.decision_function(X_test[2:5])>0).astype(int)
(clf.decision_function(X_test[2:5])>0).astype(int) * 2 - 1

y_test[2:5]

(clf.decision_function(X_test[12:15])>-2).astype(int)
(clf.decision_function(X_test[12:15]) > 2).astype(int)

for th in range(-3,7):
    print(th,(clf.decision_function(X_test[12:15])>th).astype(int))

from sklearn.metrics import roc_curve,auc,average_precision_score,precision_recall_curve

import matplotlib.pyplot as plt
%matplotlib inline

test_score = clf.decision_function(X_test)
test_score.shape
fpr,tpr,_ = roc_curve(y_test,test_score)
fpr
tpr
plt.plot(fpr,tpr)
print("AUC=",auc(fpr,tpr))

plt.plot([0,1],[0,1],linestyle="--")
plt.xlim([-0.01,1.01])
plt.ylim([0.0,1.01])
plt.ylabel("True Positive Rate (recall)")
plt.xlabel("False Positive Rate (1 - specificity)")

test_score = clf.decision_function(X_test)

precision, recall, _ = precision_recall_curve(y_test, test_score)

plt.plot(recall, precision)

plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')

test_score = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, test_score)
plt.plot(fpr, tpr, label="result")
print("result AUC = ", auc(fpr, tpr))

test_score = np.random.uniform(size=y_test.size)# もしまったくランダムなら
fpr, tpr, _ = roc_curve(y_test, test_score)
plt.plot(fpr, tpr, label="random / chance")
print("chance AUC = ", auc(fpr, tpr))

fpr, tpr, _ = roc_curve(y_test, y_test) # 完璧なら
plt.plot(fpr, tpr, label="perfect")
print("perfect AUC = ", auc(fpr, tpr))

plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc="best")
plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.01])
plt.ylabel('True Positive Rate (recall)')
plt.xlabel('False Positive Rate (1-specificity)');

test_score = clf.decision_function(X_test)
precision, recall, _ = precision_recall_curve(y_test, test_score)
plt.plot(recall, precision, label="result")

test_score = np.random.uniform(size=y_test.size) # もしまったくランダムなら
precision, recall, _ = precision_recall_curve(y_test, test_score)
plt.plot(recall, precision, label="random")

precision, recall, _ = precision_recall_curve(y_test, y_test) # 完璧なら
precision_interp = np.maximum.accumulate(precision)
plt.plot(recall, precision, marker=".",label="perfect")
plt.plot(recall, precision_interp,marker=".",label="interpolation")

plt.legend(loc="best")
plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('Recall')
plt.ylabel('Precision')

all_precision = np.interp(np.arange(0, 1.1, 0.1),
                          recall[::-1],
                          precision_interp[::-1])
AP = all_precision.mean()
all_precision
AP

def calc_AP(precision,recall):
    precision_interp = np.maximum.accumulate(precision)
    all_precision = np.interp(np.arrange(0,1.1,0.1),recall[::-1],precision_interp[::-1])
    AP = all_precision.mean()
    return AP

test_score = clf.decision_function(X_test)
precision,recall,_ = precision_recall_curve(y_test,test_score)
calc_AP(precision,recall)





# mAP of PASCAL VOC byhttp://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
APs = []
for i in range(10):

    precision, recall, _ = precision_recall_curve((y_test == i).astype(int), test_score[:,i])
    APs.append( calc_AP(precision, recall) )
APs = np.array(APs)
mAP = APs.mean()
print(APs)
print("mAP =",mAP)
