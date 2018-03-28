# 特徴量を抽出
import urllib.request
#データのダウンロード
#urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt")
data_path ="/Users/macuser/Documents/Python/scikit-learn/udemy_machine_learning/udemy_machine_learning/05_01_05/"
with open(data_path + "alice.txt","r",encoding="UTF-8") as f:
    print(f.read()[710:1400])
# テキスト用のベクトル変換用のパッケージ(単語の出現回数)
from sklearn.feature_extraction.text import CountVectorizer
txt_vec = CountVectorizer(input = "filename")
txt_vec.fit([data_path +"alice.txt"])
txt_vec.get_feature_names()[100:120]
len(txt_vec.get_feature_names())
alice_vec = txt_vec.transform([data_path + "alice.txt"])
alice_vec
alice_vec.shape
alice_vec = alice_vec.toarray()
alice_vec[0,100:120]
for word,count in zip(txt_vec.get_feature_names()[100:120],alice_vec[0,100:120]):
    print(word,count)

from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(china)

china.shape
china
# 赤の画素数をplot
histR = plt.hist(china[:,:,0].ravel(),bins=10)
# Greenの画素数をPLOT
histG = plt.hist(china[:,:,1].ravel(),bins=10)
# Blueの画素数をPLOT
histB = plt.hist(china[:,:,2].ravel(),bins=10)
import numpy as np
histRGBcat = np.hstack((histR[0],histG[0],histB[0]))
plt.bar(range(len(histRGBcat)),histRGBcat)

histRGBcat_l1 = histRGBcat / (china.shape[0]*china.shape[1])
plt.bar(range(len(histRGBcat_l1)), histRGBcat_l1);
