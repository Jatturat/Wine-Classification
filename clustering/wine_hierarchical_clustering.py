# アルゴリズム: 階層的クラスタリング(Hierarchical Clustering)
# 特徴抽出法: PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# データセット読み込み
dataset = pd.read_csv('wine_data.csv')
X = dataset.iloc[:, 1:].values
# クラスタリングは教師なし機械学習なので、従属変数（class）があっても使わないようにします。

# 標準化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# 主成分分析(PCA)で次元削減
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # 可視化するために２次元まで削減する
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# LDAは教師あり機械学習なので、従属変数がない限り利用できません。

# デンドログラムで適切なクラスター数を探す
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Classes')
plt.ylabel('Euclidean distances')
plt.show()

# 縦線が高ければ高いほどその間のデータとの関係が遠くなる
# 図中に横線から横線までの一番高い縦線に黒い横線を引いて、３縦線を切ったことで適切なクラスター数が3クラスターと分かった

# アルゴリズムに学習させる
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# クラスターの可視化
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of classes')
plt.xlabel('PC_1)')
plt.ylabel('PC_2')
plt.legend()
plt.show()

# もし下の方に横線を引いてみて7クラスターにすればどうなるか

# アルゴリズムに学習させる
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# クラスターの可視化
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'orange', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 100, c = 'pink', label = 'Cluster 6')
plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 100, c = 'black', label = 'Cluster 7')
plt.title('Clusters of classes')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.legend()
plt.show()

# 参考
# https://www.macromill.com/service/data_analysis/d004.html
