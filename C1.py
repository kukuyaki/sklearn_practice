'''
這裡要來作支援向量機的練習
使用心絞痛的數據集，透過多種特徵來預測運動是否誘發心絞痛

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
num    診斷結果                  這是最重要的目標變數，通常代表心臟疾病的嚴重程度（0=無病，1-4=疾病嚴重度）。我們的資料集只有0和1

1.根據         去推測
    能用的資料        診斷結果
    

支援向量機(Support vector machine, SVM)
支持向量分類(Support Vector Classification, SVC)
支持向量迴歸（Support Vector Regression, SVR）
模型運作概念為將資料從低維度空間中投影至高維度空間，使原本在低維度無法進行切割的資料，在高維度時能找到超平面來分開樣本


支援向量機是分類模型，一般使用線性核心SVM的時候，會在圖片上畫出一條線性的分割線，以區分兩個不同分類
但現實多數情形並沒辦法用一條直線完美分割兩種類型，這時我們就需要換別種的核心
最常見的有RBF 核 (高斯核)，幾乎可以處理各種數據

線性SVM為在二維空間上可以做到樣本的切割，當在低維度無法做到時，則會調整到高維度空間上運作，則被稱為非線性SVM。而將低維度空間轉換至高維度空間則需透過核函數(kernal function)來完成。
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 讀取與篩選資料
# ==========================================
'''
我們發現數據中，slope, ca ,thal都大部分缺失，我們直接刪除掉
其他欄位也有少許缺失值，隨後pipeline步驟時，我們再來處理
'''
df = pd.read_csv('/home/kgforsure/Documents/code/sklearn_practice/heartData/data.csv', na_values='?') #當遇到字串"?"時，將他替換成數字型態NaN, 以配合後續處理
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
其他欄位也有少許缺失值，我們用SimpleImputer（）方法，用中位數補充缺失值
將數據標準化
套接模型
'''
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # 用中位數填補，適合多數情況
    ('scaler', StandardScaler()),
    ('svm', SVC())  #SVC() 預設會使用RBF核心
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

0.847457627118644
[[34  4]
[ 5 16]]
            precision    recall  f1-score   support

0                  0.87      0.89      0.88        38
1                  0.80      0.76      0.78        21

    accuracy                           0.85        59
macro avg          0.84      0.83      0.83        59
weighted avg       0.85      0.85      0.85        59

accuracy高，混淆矩陣表現不錯
但我們的目標是找出心臟病患者，Recall的表現是我們要特別關注的
Recall: 要找到所有positive的目標，一個也不放過，目前我們只有0.76
那我們要怎麼讓recall更好呢？

這時候就可以用用看GridSearchCV！！！！！！！

但在那之前
我們目前沒有辦法可視化，因為我們的輸入特徵超多，要可視化成2D圖片，需要將"維度降低"
C2.py將會介紹PCA主成分分析！！！！！
C3.py則會介紹GridSearchCV
'''
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) # 一次顯示 P, R, F1

