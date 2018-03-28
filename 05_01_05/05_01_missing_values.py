import numpy as np
import pandas as pd
# csvファイルをインポート
data_path = "/Users/macuser/Documents/Python/scikit-learn/udemy_machine_learning/20170628ipynb/"
!cat /Users/macuser/Documents/Python/scikit-learn/udemy_machine_learning/20170628ipynb/2D_example_dame.csv

with open(data_path +"2D_example_dame.csv") as f:
    print(f.read())
# ヘッダーがない場合
df =  pd.read_csv(data_path + "2D_example_dame.csv",header=None)
# ヘッダーをつける
df_2 = pd.read_csv(data_path + "2D_example_dame.csv",names=("target","col1","col2"))
df_2
data = np.loadtxt(data_path + "2D_example_dame.csv",delimiter=",")
data
y = data[:,0].astype(int)
y

X = data[:,1:3]
X

# matplotlibで描画
import matplotlib.pyplot as plt
%matplotlib inline
#色設定
plt.set_cmap(plt.cm.Paired)

plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="k")
plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="k")
plt.xlim(-10,10)
plt.ylim(-10,10)
# NaNと外れ値の対策
# 1.NaNを除外
~np.isnan(X[:,0])

~np.isnan(X[:,1])
~np.isnan(X[:,0]) & ~np.isnan(X[:,1])
#X1にはNaNがない
X1 = X[~np.isnan(X[:,0]) & ~np.isnan(X[:,1])]
X1.shape
y1 = y[~np.isnan(X[:,0])&~np.isnan(X[:,1])]
y1.shape
# 外れ値を除外()
abs(X1[:,0])<10
abs(X1[:,1])<10
X2 = X1[((abs(X1[:,0])<10)&(abs(X1[:,1])<10))]
y2 = y1[((abs(X1[:,0])<10)&(abs(X1[:,1])<10))]
X2,X2.shape
plt.scatter(X2[:,0],X2[:,1],c=y2,s=50,edgecolors="k")

# 3.NaNを埋める(平均値で埋める)
from sklearn.preprocessing import Imputer
missing_value_to_mean = Imputer()
#学習
missing_value_to_mean.fit(X)
X.shape
#学習させたものを埋める
X_new = missing_value_to_mean.transform(X)
X_new.shape

plt.scatter(X_new[:, 0], X_new[:, 1], c=y, s=50, edgecolors='k'); # 2次元散布図でプロット
# NaNを埋める（中央値で埋める)
missing_value_to_median = Imputer(strategy="median")
missing_value_to_median.fit(X)
X_new2 = missing_value_to_median.transform(X)
pd.DataFrame(X_new2).describe()
plt.scatter(X_new2[:,0],X_new2[:,1],c=y,s=50,edgecolors="k")
