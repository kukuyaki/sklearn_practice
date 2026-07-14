#*****************************************************************************
'''
pandas的.corr功能
查看係數兩兩之間的皮爾森係數
'''

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

df = pd.read_csv("/home/kgforsure/Documents/code/sklearn_practice/wcData/player_stats.csv")
X = df[['xa','xg','shots_total','progressive_carries']]
y = df['goals']
print(X)
print(y)

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

y_test_pred = model.predict(X_test)
print(y_test_pred[:5])

print(f"訓練集 R2 分數: {model.score(X_train, y_train):.2f}")
print(f"測試集 R2 分數: {model.score(X_test, y_test):.2f}")
print(f"測試集 MAE (平均絕對誤差): {mean_absolute_error(y_test, y_test_pred):.2f}")



'''
相對於R1.py
只加上了下面程式碼
'''
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10)) 
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
