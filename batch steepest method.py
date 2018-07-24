#解説 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn.model_selection import learning_curve, validation_curve
from sklearn import linear_model, metrics, preprocessing, cross_validation #機械学習用のライブラリを利用
#from mlxtend.plotting import plot_decision_regions #学習結果をプロットする外部ライブラリを利用
import toy2

"""
#解説 2：Wineのデータセットを読み込む--------------------------------
df_wine_all=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#品種(0列、1～3)と色（10列）とプロリンの量(13列)を使用する
df_wine=df_wine_all[[0,10,13]]
df_wine.columns = [u'class', u'color', u'proline']
pd.DataFrame(df_wine)  #この行を実行するとデータが見れる
 
#解説 3：プロットしてみる------------------------------------------------------
 
x=df_wine["color"]
y=df_wine["proline"]
z=df_wine["class"]-1
plt.scatter(x,y, c=z)
plt.show
"""
X, z = make_data2(map_plot=True)
 
#解説 4：データの整形-------------------------------------------------------
#X=df_wine[["color","proline"]]
sc=preprocessing.StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
 
#解説 5：機械学習で分類する---------------------------------------------------
#clf_result=linear_model.SGDClassifier(loss="log",verbose = 1, max_iter = 100, tol = None) #loss="hinge", loss="log"
clf_result=linear_model.LogisticRegression(solver="newton-cg")
clf_result.fit(X, z)
a = clf_result.coef_[0][0]/clf_result.coef_[0][1]
b = clf_result.intercept_[0]/clf_result.coef_[0][1]
print("{:.3f}Wx + y = {:.3f}".format(a, b))#coef[0]*x+coef[1]*y+intercept=0

#train_scores, valid_scores = validation_curve(linear_model.SGDClassifier(loss="log", max_iter = 100, tol = None), X, z, "alpha",np.logspace(-7, 3, 3))
#train_scores, valid_scores = validation_curve(linear_model.LogisticRegression(solver="newton-cg"), X, z, "a", np.logspace(-7, 3, 3))
#print(train_scores)

"""
#解説 6：K分割交差検証（cross validation）で性能を評価する---------------------
scores=cross_validation.cross_val_score(clf_result, X_std, z, cv=10)
print("平均正解率 = ", scores.mean())
print("正解率の標準偏差 = ", scores.std())
 
#解説 7：トレーニングデータとテストデータに分けて実行してみる------------------
X_train, X_test, train_label, test_label=cross_validation.train_test_split(X_std,z, test_size=0.1, random_state=0)
clf_result.fit(X_train, train_label)
#正答率を求める
pre=clf_result.predict(X_test)
ac_score=metrics.accuracy_score(test_label,pre)
print("正答率 = ",ac_score)
#plotする
X_train_plot=np.vstack(X_train)
train_label_plot=np.hstack(train_label)
X_test_plot=np.vstack(X_test)
test_label_plot=np.hstack(test_label)
#plot_decision_regions(X_train_plot, train_label_plot, clf=clf_result, res=0.01) #学習データをプロット
#plot_decision_regions(X_test_plot, test_label_plot, clf=clf_result, res=0.01, legend=2) #テストデータをプロット
 
 
#解説 8：任意のデータに対する識別結果を見てみる------------------
predicted_label=clf_result.predict([1,-1])
print("このテストデータのラベル = ", predicted_label)
"""