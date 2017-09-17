# アルゴリズム: ロジスティック回帰 (Logistic Regression)
# 特徴抽出法: PCA / LDA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# データセットのインポート
dataset = pd.read_csv('wine_data.csv')

# 独立変数と従属変数の分割 
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# 学習データとテストデータの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 特徴スケーリング
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCAでデータの次元を低下する
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # 視覚化するために独立変数の数を2つにする
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# LDAでデータの次元を低下する
## from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
## lda = LDA(n_components = 2) # 視覚化するために独立変数の数を2つにする
## X_train = lda.fit_transform(X_train, y_train)
## X_test = lda.transform(X_test)

# アルゴリズムに学習させる
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# テストデータを予測する
y_pred = classifier.predict(X_test)

# 混乱行列を作って精度を測る
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
# PCAでロジスティクス回帰の結果: 精度 = 97.77%
# LDAでロジスティクス回帰の結果: 精度 = 100.00%

# データの視覚化
from matplotlib.colors import ListedColormap
# 学習データ
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression with PCA (Training set)')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
## plt.title('Logistic Regression with LDA (Training set)')
## plt.xlabel('LD_1')
## plt.ylabel('LD_2')
plt.legend()
plt.show()

# テストデータ
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression with PCA (Test set)')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
## plt.title('Logistic Regression with LDA (Test set)')
## plt.xlabel('LD_1')
## plt.ylabel('LD_2')
plt.legend()
plt.show()
