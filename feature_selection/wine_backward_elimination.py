# アルゴリズム: ロジスティック回帰とSVM
# 特徴選択法: 後ろ向き選択(Backward Elimination)

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

# 後ろ向き選択
# Given Significant Level (SL) = 0.05
# WARNING: You have to count x1,x2,... everytime after cut-off variable
#          , because xn refers to position of data in the current X_opt
import statsmodels.formula.api as sm  
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
# 決定係数（R-square）と自由度調整済みの決定係数(Adjusted R-square)を観察しながらSLを超えて一番大きなP-値の特徴を取り捨てる
# 1st round: R-sqr:0.980, Adjusted R-sqr: 0.978, cut x8 (column 7 nonflavanoid_phenols) at P value = 0.971
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
# 2nd round: R-sqr:0.979, Adjusted R-sqr: 0.978, cut x10 (column 10 hue) at P value = 0.494
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
# 3rd round: R-sqr:0.979, Adjusted R-sqr: 0.978, cut x2 (column 1 malic_acid) at P value = 0.574
X_opt = X[:, [0, 2, 3, 4, 5, 6, 8, 9, 11, 12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
# 4th round: R-sqr:0.979, Adjusted R-sqr: 0.978, cut x2 (column 2 ash) at P value = 0.329
X_opt = X[:, [0, 3, 4, 5, 6, 8, 9, 11, 12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
# 5th round: R-sqr:0.979, Adjusted R-sqr: 0.978, cut x3 (column 4 magnesium) at P value = 0.189
X_opt = X[:, [0, 3, 5, 6, 8, 9, 11, 12]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
# 6th round: R-sqr:0.979, Adjusted R-sqr: 0.978, p values of all features are lower than SL level
# Adjusted R-sqr also not decreased lower than 5th round
# Finished Backward Elimination
# Insignificant features are nonflavanoid_phenols, hue, malic_acid, ash, magnesium

# アルゴリズムに最適なデータを学習させる
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0)
X_train_opt = X_train[:, [0, 3, 5, 6, 8, 9, 11, 12]]
lr_classifier.fit(X_train_opt, y_train)

# テストデータを予測する
X_test_opt = X_test[:, [0, 3, 5, 6, 8, 9, 11, 12]] 
y_pred_lr = lr_classifier.predict(X_test_opt)

# 混乱行列を作って精度を測る
from sklearn.metrics import confusion_matrix, accuracy_score
lr_cm = confusion_matrix(y_test, y_pred_lr)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
# 後ろ向き選択でロジスティック回帰の結果: 精度 = 95.56%

# ===[ SVM ]===

# アルゴリズムに最適なデータを学習させる
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
X_train_opt = X_train[:, [0, 3, 5, 6, 8, 9, 11, 12]] 
classifier.fit(X_train_opt, y_train)

# テストデータを予測する
X_test_opt = X_test[:, [0, 3, 5, 6, 8, 9, 11, 12]] 
y_pred_svm = classifier.predict(X_test_opt)

# 混同行列を作って精度を測る
from sklearn.metrics import confusion_matrix, accuracy_score
svm_cm = confusion_matrix(y_test, y_pred_svm)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
# 後ろ向き選択でSVMの結果: 精度 = = 91.11%


# 4つ以上の特徴が残っているため可視化ができません

# 参考
# https://www.techcrowd.jp/machinelearning/dimensions/