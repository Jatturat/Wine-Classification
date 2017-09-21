# アルゴリズム: k-means++ クラスタリング(K-Means++ Clustering)
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

# elbow法とWCSS（within-cluster sum of squares）で適切なクラスター数を探す
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# 肘（elbow）のところにWCSSが3なので適切なクラスター数（ｋ）が３クラスターと分かった

# アルゴリズムに学習させる
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# クラスターの可視化
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of wine classes')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.legend()
plt.show()

# 参考
# http://rindalog.blogspot.com/2016/08/blog-post_24.html