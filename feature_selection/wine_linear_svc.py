# アルゴリズム: ロジスティック回帰とSVM
# 特徴選択法: 線形SVR(Linear SVR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# データセット読み込み
dataset = pd.read_csv('wine_data.csv')

# 独立変数と従属変数の分割
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# 学習データとテストデータの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 主成分分析や線形判別分類などの特徴抽出法を使わずにどの特徴が重要か重要ではないかを明確にするために特徴選択法を使う

# 線形SVR
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y) # dual=False is for when n_samples > n_features (dataset has more rows than columns)
model = SelectFromModel(lsvc, prefit=True)
X_train_opt = model.transform(X_train)
X_test_opt = model.transform(X_test)

# By comparing dataset, X_train and X_train_opt, I found that these are the features chosen by Linear SVR
# High impact features: alcohol, flavanoids, color_intensity, hue, OD280_OD315, proline
# And these features are cut by Linear SVR
# Low impact features: malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, nonflavanoid_phenols, proanthocyanins

# ===[ ロジスティック回帰 ]===

# アルゴリズムに最適なデータを学習させる
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0)
lr_classifier.fit(X_train_opt, y_train)

# テストデータを予測する
y_pred_lr = lr_classifier.predict(X_test_opt)

# 混乱行列を作って精度を測る
from sklearn.metrics import confusion_matrix, accuracy_score
lr_cm = confusion_matrix(y_test, y_pred_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
# 線形SVRでロジスティック回帰の結果: 精度 = 97.78% 

# ===[ SVM ]===

# アルゴリズムに最適なデータを学習させる
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_opt, y_train)

# テストデータを予測する
y_pred_svm = classifier.predict(X_test_opt)

# 混同行列を作って精度を測る
from sklearn.metrics import confusion_matrix, accuracy_score
svm_cm = confusion_matrix(y_test, y_pred_svm)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
# 線形SVRでSVMの結果: 精度 = 95.56% 


# 4つ以上の特徴が残っているため可視化ができません

# 参考
# https://www.techcrowd.jp/machinelearning/dimensions/