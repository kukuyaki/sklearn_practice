'''
回到C1.py的問題
我們要提升recall，可參考以下方法

1. 使用 GridSearchCV 調參 (自動化優化)
2. 調整類別權重 (最推薦的策略)
3. 調整分類閾值 (Threshold Moving)

我們將嘗試使用第一種方法
GridSearchCV是類似排列組合的方法，根據我們提供的變數，挑選各種組合，去訓練多次模型

比如
核心          主成分數量     SVM_C
    linear          2           0.1
    RBF             3           0.01

grid = GridSearchCV(model, param_grid, cv=5, scoring='recall')
cv代表我們要將原始數據切成幾份，跑幾次交叉驗證
scoring代表要用哪一個指標去決定最後的最佳模型

我們從這行程式碼中發現，GridSearchCV()步驟其實是直接取代原本的model訓練model.fit()的。


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


'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 讀取與篩選資料
# ==========================================

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

model = Pipeline([  #pipeline裡面的順序很重要，不能隨意調整順序
    ('imputer', SimpleImputer(strategy='median')), # 用中位數填補，適合多數情況
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC())  #SVC() 預設會使用RBF核心
])
# ==========================================
# 4. 模型優化與調參GridSearchCV vs RandomizedSearchCV
# ==========================================
'''
我們有四個變數
svm_c：錯誤分類的懲罰程度
svm_gamma：單個資料點對邊界計算的影響力範圍
svm_kernel：使用的核心
pca__n_components：主成分數量

分別有4,4,2,4個選擇
表示會跑4*4*2*4 = 128個模型
且cv=5
表示每一個模型會跑5次，然後取平均分數
也就是總共會跑128*5次，所以使用GridSearchCV時會需要點時間
'''
# 定義你想搜尋的參數空間
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [1, 0.1, 0.01, 'scale'],
    'svm__kernel': ['rbf', 'linear'],
    'pca__n_components': [2, 3, 5, 7]
}

# 使用 GridSearchCV
model = GridSearchCV(model, param_grid, cv=5, scoring='recall') # 這裡把目標設為提升 recall
model.fit(X_train, y_train)

print("最佳參數:", model.best_params_)
# ==========================================
# 5. 訓練資料，fit
# ==========================================
'''
被GridSearchCV取代
'''
# ==========================================
# 6. 實際預測範例
# ==========================================
y_pred = model.predict(X_test)
# ==========================================
# 7. 評估結果
# ==========================================
'''
最後的recall確實有提升
從0.76變成0.81了
最佳參數: {'pca__n_components': 3, 'svm__C': 1, 'svm__gamma': 1, 'svm__kernel': 'linear'}
'''
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) # 一次顯示 P, R, F1

# ==========================================
# 8. 畫圖
# ==========================================
'''
因為最後的pca主成分不一定是2維，我們就先不畫圖了

複習一下
C1.py學習了支援向量機的使用
C2.py為了降低維度，學了PCA主成分分析
C3.py為了優化模型，學了GridSearchCV

回歸和分類的模型都學過了，你會發現sklearn真的很強大
我們其實只是在換模型而已，也就表示要學會機器學習，比起程式碼撰寫，更重要的是知道你要為了什麼目的而使用什麼技術去處理你面對的資料集。

如果你是想要去研發更多的工具，也就是發明更新更好的模型，你要學習我們使用的模型底下的數學原理
如果你是想要應用這些工具去處理你面對的問題，你就要培養對現有模型的認知和熟悉度

一個是研究導向，向下挖掘
一個是應用導向，橫向擴展
好好思考你是為了什麼而去學習機器學習

下一步，我們來學習非監督式學習
'''