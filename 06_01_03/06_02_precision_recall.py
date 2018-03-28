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
train_index,test_index = next(ss.split(X,y))

X_train,X_test = X[train_index],X[test_index]
y_train,y_test = y[train_index],y[test_index]

from sklearn import linear_model
clf = linear_model.LogisticRegression()

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_test,y_pred)
cmat = confusion_matrix(y_test,y_pred)
cmat

cmat.sum()
cmat.diagonal().sum()
cmat.diagonal().sum() / cmat.sum()

TP = cmat[0,0]
TN = cmat[1,1]
FP = cmat[1,0]
FN = cmat[0,1]
TP,TN,FP,FN

from sklearn.metrics import classification_report
# digits(小数点)
print(classification_report(y_test,y_pred,digits=4))
#recallの説明
#再現率
recall_0 = TP /(TP+FN)

# = 46/46+1 class 0 recall 再現率,
#           sensitivity 感度,
#           True Positive rate (TPR)

recall_0

#precision
#適合度
precision_0 = TP /(TP+FP)

# 46/(46+4) class 0 precision 適合度、精度

precision_0

recall_1 = TN /(FP + TN)
# 63 /(63+4) class 1 recall,
#            specificity 特異度 vs sensitivity 感度

specificity = recall_1
recall_1

# False Positive Rate
FP /(FP + TN)
# False positive rate(FPR) = 1 - specificity

precision_1 = TN/(TN+FN)
# 63/(63+1) class 1 precision

precision_1

# f1score
f1_0 = 2*recall_0*precision_0/(recall_0 + precision_0)
# 2/(1/recall_0 + 1/precision_0)
f1_0

f1_1 = 2*recall_1*precision_1/(recall_1 + precision_1)
# 2/(1{recall_1 + 1/precision_!})

f1_1

from sklearn.metrics import f1_score

f1_score(y_test,y_pred,pos_label=0)
f1_score(y_test,y_pred,pos_label=1)

# betaを1に設定するとf1_scoreの結果と一致
from sklearn.metrics import fbeta_score
fbeta_score(y_test,y_pred,beta=1,pos_label=0)
fbeta_score(y_test,y_pred,beta=1,pos_label=1)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test,y_pred,beta=1)

# avgの説明
(precision_0 + precision_1)/2
(precision_0*47 + precision_1*67)/114

# 10class problem

from sklearn.datasets import load_digits
data = load_digits()

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

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred,digits=4))
