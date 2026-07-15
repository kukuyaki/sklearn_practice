'''
使用心絞痛的數據集，透過多種特徵來預測運動是否誘發心絞痛
這次我們要來將高維度數據先轉成低維度數據，再餵給模型訓練
也就是說PCA是訓練前的步驟

PCA的原理要讀懂線性代數後比較好理解，努力讀線性代數吧！！！！！！

一、 PCA 的主要目的
1.資料降維（Dimensionality Reduction）：
將數十、數百個特徵減少到幾個關鍵指標，大幅降低模型的複雜度。

2.消除共線性（Multicollinearity）：
當原始特徵之間高度相關（例如：血壓與膽固醇可能隨年齡共同變化），模型會因為重複資訊而受到干擾；PCA 會將這些相關特徵重組為「不相關」的主成分，解決此問題。

3.過度擬合（Overfitting）防護：
減少特徵維度有助於消除雜訊，讓模型專注於資料中最主要的結構，進而提升泛化能力。

4.視覺化（Visualization）：
將高維資料壓縮至 2D 或 3D 平面，讓我們能用肉眼觀察模型是如何劃分資料的（如你現在繪製的圖表）。

我們這次使用PCA的目的是為了第四點的視覺化。

若目標是「解釋性 (Interpretation)」：
    例如你必須告訴醫生「具體是因為哪一個指數變高才導致心臟病」，則 不建議使用 PCA，因為 PCA 產生出的主成分（如 PC1, PC2）已失去原始欄位的物理意義。


age	年齡	                    患者的年齡
sex	性別	                    1 = 男性, 0 = 女性
cp	胸痛類型	                 胸痛的型態 (Chest Pain Type)
trestbps	 靜息血壓	         患者休息時的收縮壓 (單位：mmHg)
chol	血清膽固醇	             血清中的總膽固醇濃度 (單位：mg/dl)

fbs	空腹血糖	                 是否 > 120 mg/dl (1 = 是, 0 = 否)

restecg	靜息心電圖	             休息時的心電圖結果 (Resting Electrocardiographic Results)
thalach	最大心率	             運動時達到的最大心率 (Maximum Heart Rate Achieved)

exang	運動誘發心絞痛	          運動是否引起胸痛 (1 = 是, 0 = 否)
oldpeak	運動引起的 ST 段壓低	  與休息狀態相比，運動誘發的 ST 段壓低程度 (指標心肌受損程度)

slope  ST 段斜率                 運動過程中 ST 段的傾斜坡度（指標心肌供血狀態）。
ca    冠狀動脈鈣化數              使用螢光檢查看到的血管鈣化數量（0-3），與血管阻塞風險高度相關。
thal   地中海貧血狀態             這裡通常指「心肌灌注檢查」的結果，是一個很強的病理指標。
num    診斷結果                  這是最重要的目標變數，通常代表心臟疾病的嚴重程度（0=無病，1-4=疾病嚴重度）。


'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 讀取與篩選資料
# ==========================================

df = pd.read_csv('/home/kgforsure/Documents/code/sklearn_practice/heartData/data.csv', na_values='?') #當遇到字串"?"時，將他替換成數字型態NaN
df = df.drop(columns=['slope','ca','thal'])
X = df.drop(columns=['num'])
y = df['num']

# ==========================================
# 2. 切分資料訓練與測試集
# ==========================================
X_train, X_test, y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. 建立 Pipeline，選擇要用的資料前處理工具
# ==========================================
'''
程式碼相對於C1.py只有更改這裡以及後面畫圖的部分
'''
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # 用中位數填補，適合多數情況
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('svm', SVC())
    # ('svm', SVC(kernel='linear'))
])
# ==========================================
# 4. 模型優化與調參GridSearchCV vs RandomizedSearchCV
# ==========================================
'''
先無視
'''
# ==========================================
# 5. 訓練資料，fit
# ==========================================
model.fit(X_train,y_train)
# ==========================================
# 6. 實際預測範例
# ==========================================
y_pred = model.predict(X_test)
# ==========================================
# 7. 評估結果
# ==========================================
# y_test: 真實標籤, y_pred: 模型預測標籤
'''
因為這次是分類問題
我們用以下指標評估模型

以下程式碼有改動
新增pca貢獻度
PCA 貢獻度: [0.25693808 0.13172472]
代表主成分1和主成分2分別抓取原始數據多少比例的資訊
合計約0.38，且表現還不錯
這是好事情，表示使用0.38的資訊就有不錯或甚至更好的解果
'''
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) # 一次顯示 P, R, F1

pca = model.named_steps['pca']
print(f"PCA 貢獻度: {pca.explained_variance_ratio_}")


# ==========================================
# 8. 畫圖
# ==========================================
x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1 # 這裡僅用前兩個欄位作邊界範圍參考
X_pca = model.named_steps['pca'].transform(model.named_steps['scaler'].transform(model.named_steps['imputer'].transform(X_test)))
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

Z = model.named_steps['svm'].predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("SVM Decision Boundary (After PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


'''
下一關，我們將會為了有更好的recall，而使用GridSearchCV
'''