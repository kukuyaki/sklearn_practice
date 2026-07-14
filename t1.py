#*****************************************************************************
'''
這是一個根據ML_python_guide.py寫出來的示範程式碼
目標是根據球員資料集合得出一些相對關係
先看資料有哪些特徵，才能決定要使用哪些特徵去預測哪些特徵，也就是要拿這個數據集做什麼

球員編號 (player_id)
球員姓名 (player)
球隊 (team)
位置組別 (position_group)
比賽編號 (match_id)

預期進球值 (xg)
進球數 (goals)
射門次數 (shots_total)
預期助攻值 (xa)
關鍵傳球 (key_passes)
推進傳球 (progressive_passes)
推進運球 (progressive_carries)
壓迫次數 (pressures)
成功鏟球 (tackles_won)
攔截次數 (interceptions)
解圍次數 (clearances)
上場時間 (minutes)

可看出來
前半部是字串數據：說不定可以當我們的預測目標y，使用前要先轉換成數字形式（oneHotEncoder），可用於分類問題的預測目標
後半部是數字數據:說不定可以當我們的輸入X，也可以當預測目標y

前半部再這份資料集當中，不太適合當成預測目標y，因為樣本有點少，總共1996筆資料，每個球員可能只分到1筆，每個國家可能只分到10筆，每個比賽可能只分到20筆，因此訓練效果會很差
真的想預測前半部的話，比起國家，我們可以把多個國家合併，變成先以洲為目標，這樣每一洲的數據量更大，更好預測

但我們還是先只使用後半部的資料，我們可以想像幾種不錯的搭配。如下，兩種都是回歸問題，得到的輸出是連續的數值

迴歸 (Regression) 處理的是連續型數據 (Continuous Data)。
分類 (Classification) 處理的是離散型數據 (Categorical/Discrete Data)。

1.根據         去推測
    預期助攻值        進球數
    預期進球值
    射門次數
    推進運球

2.根據         去推測
    解圍次數        關鍵傳球
    成功鏟球
    攔截次數
    推進傳球


我們先以第一組為目標

'''
#*****************************************************************************

#import 需要的工具
#數據處理
#sklearn前處理
#sklearn模型
#sklearn評分
#sklearn其他工具
#畫圖
#儲存model工具 import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 讀取與篩選資料
# ==========================================
df = pd.read_csv("/home/kgforsure/Documents/code/sklearn_practice/wcData/player_stats.csv")
X = df[['xa','xg','shots_total','progressive_carries']]
y = df['goals']
print(X)
print(y)

# ==========================================
# 2. 切分資料訓練與測試集
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ==========================================
# 3. 建立 Pipeline，選擇要用的資料前處理工具
# ==========================================
'''
我們可以先把X的數字欄位作標準化，再用我們要用的模型，先用用看線性回歸模型
'''
model = Pipeline(steps=[
    ('preprocessor', StandardScaler()
        ),
    ('classifier', LinearRegression(
        n_jobs=-1              # 使用所有 CPU 核心加速運算
    ))
])
# ==========================================
# 4. 模型優化與調參GridSearchCV vs RandomizedSearchCV
# ==========================================
'''
這步驟是要作模型的參數的排列組合的窮舉的，線性回歸沒有太多參數可以調整，這步驟就先跳過
比如隨機森林模型，可以調整深度、小模型選擇等參數，這時就可以用這一個步驟去排列各種參數組合
'''
# ==========================================
# 4. 訓練資料，fit
# ==========================================
'''
你會看到4種不同的函式
model.fit()
model.transform()
model.fit_transform()
model.predict()

fit 只在訓練集上使用，讓model去學習
transform 在各種情況都會用到，用學到的特徵去將舊的資料改成新資料

fit 只計算資料的平均值（Mean）和標準差（Standard Deviation），並把它們記在記憶體中。
transform 利用剛剛 fit 學到的平均值和標準差，把資料公式化轉換（縮放）成全新的數值。

predict 輸入輸入特徵X，透過訓練好的模型，去吐出我們要的預測數據y

所以我們先訓練，要用fit
然後predict
'''
model.fit(X_train, y_train)
# ==========================================
# 5. 實際預測範例
# ==========================================
y_test_pred = model.predict(X_test)
print(y_test_pred[:5])
# ==========================================
# 6. 評估結果
# ==========================================
'''
📈 分類指標 (Classification)針對：類別預測 (如：有無進球、輸贏)
    Accuracy (準確度)：整體預測正確的比例。（資料不平衡時參考價值低）
    Precision (精確率)：預測為正類中，正確的比例。（重視：不誤報）
    Recall (召回率)：真實正類中，被找出的比例。（重視：不漏報）
    F1-Score：精確率與召回率的調和平均。（綜合指標）ROC-AUC：模型分辨正負類的能力。（越接近 1 越好）

📉 迴歸指標 (Regression)針對：數值預測 (如：進球數、預期進球值)
    R² Score (決定係數)：解釋變異的能力。（範圍：$0 \sim 1$，越高越好）
    MAE (平均絕對誤差)：預測值與真實值的平均差距。（單位與目標一致，易解釋）
    MSE (均方誤差)：誤差平方的平均。（會放大較大的誤差，對異常值敏感）
    RMSE (均方根誤差)：MSE 開根號。（兼具平方特性與原始單位）

因為我們是回歸問題，所以我們用回歸的指標去評分
'''
print(f"訓練集 R2 分數: {model.score(X_train, y_train):.2f}")
print(f"測試集 R2 分數: {model.score(X_test, y_test):.2f}")
print(f"測試集 MAE (平均絕對誤差): {mean_absolute_error(y_test, y_test_pred):.2f}")



