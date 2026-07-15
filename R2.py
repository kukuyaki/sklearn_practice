#*****************************************************************************
'''
這是一個根據ML_python_guide.py寫出來的示範程式碼
上次我們成功使用回歸模型得到預測結果，但使用的輸入特徵和輸出預測特徵是我們用腦瓜子想出來的，但透過經驗想出來的不一定對
所以為了更好的挑選要使用的輸入輸出，我們可以用以下方法

1.我們可以試試看各種排列組合窮舉
2.先看相關係數矩陣 (Correlation Matrix)
3.使用有些模型給的自動化的特徵選擇 (Feature Selection)

1很可能導致過擬合，且會跑超慢

2則可以透過pandas的.corr功能查看係數兩兩之間的皮爾森係數，也就是相關性
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

3則是某些模型預測完後會有特徵重要性的查看功能，比如randomforest的.feature_importances_，可以在第一次訓練模型後，根據特徵重要性的結果去刪減不重要的特徵，再去作訓練
rf = RandomForestRegressor()
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar')

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


我們來試試看第一種方法，只使用下半部分的連續型數字數據，排列各種組合後輸出評分報告，我們就可以從報告中看出端倪。
我們直接把上次的程式碼改成一個函式，liner_report(特徵輸入, 特徵輸出)，我們讓這個函式回傳一個report

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
import random
import itertools
import csv
import os
import time

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


def liner_report(df,a,b):
    # df = pd.read_csv("/home/kgforsure/Documents/code/sklearn_practice/wcData/player_stats.csv")
    # X = df[['xa','xg','shots_total','progressive_carries']]
    # y = df['goals']
    X = df[a]
    y = df[b]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(steps=[
        ('preprocessor', StandardScaler()
            ),
        ('classifier', LinearRegression(
            n_jobs=-1              # 使用所有 CPU 核心加速運算
        ))
    ])
    model.fit(X_train, y_train)
    os.system('cls' if os.name == 'nt' else 'clear')
    end_time = time.time()
    print(end_time - start_time)

    y_test_pred = model.predict(X_test)
    if model.score(X_train, y_train) > 0.5 and model.score(X_test, y_test) >0.5 and mean_absolute_error(y_test, y_test_pred) < 0.3:
        print(f"{len(a)}")
        print(f"{a}")
        print(f"{b}")
        print(f"訓練集 R2 分數: {model.score(X_train, y_train):.2f}")
        print(f"測試集 R2 分數: {model.score(X_test, y_test):.2f}")
        print(f"測試集 MAE (平均絕對誤差): {mean_absolute_error(y_test, y_test_pred):.2f}")

    # with open("example.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{len(a)}\n")
    #     f.write(f"{a}\n")
    #     f.write(f"{b}\n")
    #     f.write(f"訓練集 R2 分數: {model.score(X_train, y_train):.2f}\n")
    #     f.write(f"測試集 R2 分數: {model.score(X_test, y_test):.2f}\n")
    #     f.write(f"測試集 MAE (平均絕對誤差): {mean_absolute_error(y_test, y_test_pred):.2f}\n\n")
    #     if model.score(X_train, y_train) > 0.5 or model.score(X_test, y_test) >0.5 or mean_absolute_error(y_test, y_test_pred) < 0.3:
    #         f.write("sogood")

    with open("feature_report.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if model.score(X_train, y_train) > 0.5 and model.score(X_test, y_test) >0.5 and mean_absolute_error(y_test, y_test_pred) < 0.3:
            writer.writerow([len(a), a,b,model.score(X_train, y_train),model.score(X_test, y_test),mean_absolute_error(y_test, y_test_pred),"sogood"])
        else:
            writer.writerow([len(a), a,b,model.score(X_train, y_train),model.score(X_test, y_test),mean_absolute_error(y_test, y_test_pred)])

df = pd.read_csv("/home/kgforsure/Documents/code/sklearn_practice/wcData/player_stats.csv")

df = df.drop(columns=['team', 'player_id', 'player', 'match_id', 'position_group'])
nofc = df.shape[1]  #總共17個特徵，刪掉五個，剩下12個
all_fea = df.columns.tolist()
start_time = time.time()
for j in range(len(all_fea)):
    b=all_fea[j]
    ax = [x for x in all_fea if x != b]
    for i in range(1,nofc):  #從1到11, 原本有12個特徵，留一個給輸出特徵，剩下11個 此為輸入特徵的數量，輸出特徵我們設定為只有一個就好
        a = itertools.combinations(ax, i)
        for combo in a:
            combo = list(combo)
            liner_report(df,combo,b)

'''
執行時間達到驚人的265秒，超久，這就是為什麼我們不用窮舉的
而且仔細觀察輸出的csv數據，可以看出表現最好的輸入輸出選擇普遍會和我們用經驗累積的判斷相似

舉例
輸入['goals', 'shots_total', 'xa', 'key_passes', 'progressive_passes', 'progressive_carries', 'pressures', 'tackles_won', 'interceptions', 'clearances', 'minutes']	
輸出 xg	

0.638596081865663	
0.520427126242759	
0.0661499510771998

xg是預計進球數，本來就是別人用各種數據去算出來的，我們所使用的輸入多半也是別人使用的數據中的一部分，所以這有一點點數據洩漏的感覺



下一關是R3.py
趕快去看一下pandas的.corr功能實作
'''