import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. 讀取與處理
df = pd.read_csv('/home/kgforsure/Downloads/world cup datasets/player_stats.csv')

# 假設你已經選擇好了要分類的隊伍 (例如前兩大)
top_teams = df['team'].value_counts().nlargest(2).index
df_filtered = df[df['team'].isin(top_teams)].copy()
tech_cols = ['pressures', 'progressive_passes', 'progressive_carries']
for col in tech_cols:
    df_filtered[col] = df_filtered[col] / (df_filtered['minutes'] / 90)
# 定義輸入與輸出
X = df_filtered.drop(columns=['team', 'player_id', 'player', 'match_id', 'position_group','minutes'])
y = df_filtered['team']

# 自動偵測數值欄位 (過濾掉文字型別)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# 2. 建立 Pipeline
# 隨機森林其實不需要 StandardScaler，但保留它無妨，或可選擇註解掉
model = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])),
    ('classifier', RandomForestClassifier(
        n_estimators=100,      # 建立 100 棵決策樹
        max_depth=5,           # 限制樹深度以防止過擬合
        random_state=42,       # 固定亂數種子，確保結果可重現
        n_jobs=-1              # 使用所有 CPU 核心加速運算
    ))
])

# 3. 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 訓練
model.fit(X_train, y_train)

# 5. 評估結果
print(f"訓練集準確率: {model.score(X_train, y_train):.2f}")
print(f"測試集準確率: {model.score(X_test, y_test):.2f}")
print("\n詳細報告:\n", classification_report(y_test, y_pred=model.predict(X_test)))

# 6. 查看特徵重要性 (這是隨機森林的靈魂！)
importances = model.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=numeric_features).sort_values(ascending=False)

print("\n模型認為區分這兩國最重要的三個指標：")
print(feat_imp.head(3))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_filtered, x='team', y='pressures')
plt.title('Argentina vs Morocco: Pressure Distribution')
plt.show()