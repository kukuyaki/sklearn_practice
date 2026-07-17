
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
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
    ('ridge', make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0)))
]

# 2. 定義堆疊模型 (Meta Learner)
# 用邏輯回歸來學習如何組合這三個模型的預測結果
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(max_depth=5),
    n_jobs=-1,
    cv=5, # 使用 5 折交叉驗證來避免 Meta Learner 過擬合
    verbose=1
)

# 3. 訓練
print("開始訓練 Stacking 模型...")
stack_model.fit(X_train, y_train)

# 4. 評估
y_pred_stack = stack_model.predict(X_test)
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
    # 這裡的 sample 僅用於視覺化展示，不會影響 stack_model 的訓練結果
    acc = accuracy_score(y_test, model.predict(X_test))
    model_accs.append(acc)

sns.barplot(x=model_names, y=model_accs, ax=ax2, palette='viridis')
ax2.set_title('Base Learners Accuracy Comparison')
ax2.set_ylabel('Accuracy')

# 3. Meta-Learner 的特徵重要性觀察 (修正後)
# 隨機森林不使用 coef_，我們改用 feature_importances_
ax3 = plt.subplot(2, 1, 2)
importances = stack_model.final_estimator_.feature_importances_

# 隨機森林作為 Meta-Learner 時，它會根據每個基礎模型的預測結果(機率或類別)進行判斷
# 這些特徵的數量等於 (基礎模型數量 * 類別數量)
sns.barplot(x=importances, y=[f"{m}_feature_{i}" for m in model_names for i in range(len(importances)//len(model_names))], ax=ax3)
ax3.set_title('Meta-Learner Feature Importance (Which base model output is most important?)')
ax3.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig(f'model_evaluation_T4_2.png', dpi=300, bbox_inches='tight')
plt.show()

'''
我們使用StackingClassifier集成學習的方式，套用多個模型去預測
但不但每一個小模型效果都不好，最後預測出來的結果也不好

即使我們使用了更強大的模型，還是無法從2手機體訊息判斷出品牌，這代表這些2手賣價的特徵所持有的資訊量，對於判斷品牌是不足夠的
如果真的想要判斷品牌，可能要有其他特徵



嘗試一：使用邏輯斯函數當final_estimator
Stacking 模型準確率: 0.2859
              precision    recall  f1-score   support

         0.0       0.48      0.56      0.52     28659
         1.0       0.23      0.16      0.19     28615
         2.0       0.21      0.25      0.23     28648
         3.0       0.29      0.53      0.37     28655
         4.0       0.26      0.25      0.25     28329
         5.0       0.23      0.07      0.11     28491
         6.0       0.21      0.17      0.19     28603

    accuracy                           0.29    200000
   macro avg       0.27      0.29      0.27    200000
weighted avg       0.27      0.29      0.27    200000

嘗試一：使用隨機森林函數當final_estimator
Stacking 模型準確率: 0.2836
              precision    recall  f1-score   support

         0.0       0.48      0.57      0.52     28659
         1.0       0.22      0.15      0.18     28615
         2.0       0.20      0.24      0.22     28648
         3.0       0.30      0.52      0.38     28655
         4.0       0.25      0.25      0.25     28329
         5.0       0.22      0.08      0.12     28491
         6.0       0.21      0.17      0.19     28603

    accuracy                           0.28    200000
   macro avg       0.27      0.28      0.26    200000
weighted avg       0.27      0.28      0.27    200000

'''

