'''
用RandomForestClassifier來作分類問題


能用計算與邏輯規則處理的，就不要讓模型去學習；能用一個模型處理的，就不要拆成兩個模型
回歸模型後可以用後處理去改成分類
對於回歸問題，我們不需要只為了要分類而去再跑一個分類模型

那我們這次要來設定什麼目標好呢

我們來預測品牌
要預測brand時，記得要把model特徵刪掉，因為他和品牌強綁定
OS_type也要刪掉，因為只有蘋果會用ios
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 讀取與篩選資料
# ==========================================
'''
文字轉成OrdinalEncoder：
    condition
    seller_type
    brand
    model
'''
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
model = RandomForestClassifier(
    n_estimators=100,        # 先從 100 棵樹開始，不要一開始就設 500-1000
    max_depth=15,           # 限制深度是防止過擬合且加速訓練最有效的方法
    n_jobs=-1,              # **最重要**：啟用所有核心進行平行運算
    max_samples=0.5,        # 每次只用 30% 的數據訓練一棵樹，顯著加速且通常能減少過擬合
    random_state=42,
    verbose=1    #進度條
)
model.fit(X_train,y_train)
print("模型訓練完畢")


y_pred = model.predict(X_test)
print("模型預測完畢")

# ==========================================
# 7. 評估結果
# ==========================================
print(f"分類模型準確率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\n分類報告 (Classification Report):")
print(classification_report(y_test, y_pred))

# --- 修正後的視覺化部分 (混淆矩陣) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# --- 1. 畫出特徵重要性 (ax1) ---
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Feature Importances')
ax1.set_ylabel('Importance Score')
ax1.set_xlabel('Features')

# --- 2. 畫出混淆矩陣 (ax2) ---
cm = confusion_matrix(y_test, y_pred)
# 取得品牌的類別名稱 (假設你的 encoder 已經 fit 過)
brand_names = encoder.categories_[list(cat_cols).index('brand')]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=brand_names,
            yticklabels=brand_names)

ax2.set_title('Brand Prediction Confusion Matrix')
ax2.set_xlabel('Predicted Brand')
ax2.set_ylabel('Actual Brand')

# 自動調整佈局，避免標籤重疊
plt.tight_layout()
plt.savefig(f'model_evaluation_T3.png', dpi=300, bbox_inches='tight')

plt.show()

'''
效果並不好
蘋果相較於其他品牌，容易分辨
我們有7個預測可能，隨機猜的話每個有13％左右的機率，我們的有accuracy0.22的表現，算是稍微能判斷

這個結果可能表示了什麼
1.單純模型很差
2.各品牌之間差異，不以硬體數據為分界

會不會品牌之間主要的差異在於維修服務、軟體介面、相容性等等呢

我們因為資料不夠無法測試第二點
但我們可以透過改變模型和訓練策略去驗證第一點

T4.py我們將強化我們模型的能力，使用比隨機森林更強大的模型
'''