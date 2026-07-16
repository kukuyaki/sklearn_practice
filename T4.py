
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/home/kgforsure/Documents/code/sklearn_practice/use_phone/used_phone_price_prediction_1M.csv')
df['diff_price'] = df['original_price'] - df['resale_price']
df['diff_price_ratio'] = df['resale_price'] / df['original_price']
df = df.drop(columns=['resale_price'])

cat_cols = df.select_dtypes(include=['object', 'string']).columns
encoder = OrdinalEncoder()
df[cat_cols] = encoder.fit_transform(df[cat_cols])
pre_drop = ['diff_price', 'brand','model','os_type']
X = df.drop(columns=pre_drop)
y= df['brand']
#將資料儲存空間變小一點，跑的比較快
df = df.astype({col: 'float32' for col in df.select_dtypes(include=['float64']).columns})
df = df.astype({col: 'int32' for col in df.select_dtypes(include=['int64']).columns})
# ==========================================
# 2. 切分資料訓練與測試集
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("資料整理完畢")

# 1. 定義基礎模型 (Base Learners)
# 這裡使用不同的算法，讓它們從不同角度「看」資料
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)),
    ('ridge', RidgeClassifier(alpha=1.0))
]

# 2. 定義堆疊模型 (Meta Learner)
# 用邏輯回歸來學習如何組合這三個模型的預測結果
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    n_jobs=-1,
    cv=5, # 使用 5 折交叉驗證來避免 Meta Learner 過擬合
    verbose=1
)

# 3. 訓練
print("開始訓練 Stacking 模型...")
stack_model.fit(X_train, y_train)

# 4. 評估
y_pred_stack = stack_model.predict(X_test)
print(f"Stacking 模型準確率: {accuracy_score(y_test, y_pred_stack):.4f}")
print(classification_report(y_test, y_pred_stack))

再加上畫圖
請多一點圖形

# 4. 評估
y_pred_stack = stack_model.predict(X_test)
print(f"Stacking 模型準確率: {accuracy_score(y_test, y_pred_stack):.4f}")

# --- 多圖繪製開始 ---
fig = plt.figure(figsize=(20, 15))

# 1. 混淆矩陣
ax1 = plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred_stack)
brand_names = encoder.categories_[list(cat_cols).index('brand')]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=brand_names, yticklabels=brand_names)
ax1.set_title('Stacking Model Confusion Matrix')

# 2. 各基礎模型的準確率比較
ax2 = plt.subplot(2, 2, 2)
model_names = [e[0] for e in estimators]
model_accs = []
for name, model in estimators:
    model.fit(X_train.sample(50000, random_state=42), y_train.sample(50000, random_state=42))
    acc = accuracy_score(y_test, model.predict(X_test))
    model_accs.append(acc)

sns.barplot(x=model_names, y=model_accs, ax=ax2, palette='viridis')
ax2.set_title('Base Learners Accuracy Comparison')
ax2.set_ylabel('Accuracy')

# 3. Meta-Learner 的係數（權重）觀察
# 邏輯回歸的係數代表它對每個基礎模型輸出的信心程度
ax3 = plt.subplot(2, 1, 2)
coefs = stack_model.final_estimator_.coef_
# 將係數畫出，了解 Meta-Learner 偏好哪個模型
sns.heatmap(coefs, annot=True, cmap='coolwarm', ax=ax3,
            xticklabels=[f"{m}_class_{i}" for m in model_names for i in range(len(brand_names))],
            yticklabels=brand_names)
ax3.set_title('Meta-Learner Weights (Which model does it trust?)')

plt.tight_layout()
plt.savefig(f'model_evaluation_T4.png', dpi=300, bbox_inches='tight')
plt.show()


'''
因為強化了不少要跑久一點
還沒跑出來但要下課了 TODO
'''