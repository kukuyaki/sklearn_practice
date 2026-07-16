'''
我們來測試拿掉original_price後會如何
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,r2_score, mean_absolute_error
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
pre_drop = ['diff_price', 'diff_price_ratio','original_price']
X = df.drop(columns=pre_drop)
for single_target in ['diff_price','diff_price_ratio']:
    y= df[single_target]
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
    model = RandomForestRegressor(
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
    print(f"預測目標: {single_target}")
    print(f"訓練集 R2 分數: {model.score(X_train, y_train):.2f}")
    print(f"測試集 R2 分數: {model.score(X_test, y_test):.2f}")
    print(f"測試集 MAE (平均絕對誤差): {mean_absolute_error(y_test, y_pred):.2f}")

    # ==========================================
    # 8. 畫圖
    # ==========================================
    # 畫出特徵重要性
    # 使用 subplots 建立 1 列 2 欄的畫布 (figsize 控制整體大小)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- 第一張圖：特徵重要性 (放在 ax1) ---
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Feature Importances')
    ax1.set_ylabel('Importance')
    ax1.set_xlabel('Features')

    # --- 第二張圖：預測 vs 真實 (放在 ax2) ---
    sample_idx = np.random.choice(len(y_test), 5000, replace=False)
    y_test_sample = y_test.iloc[sample_idx]
    y_pred_sample = y_pred[sample_idx]

    ax2.scatter(y_test_sample, y_pred_sample, alpha=0.3, color='blue', s=10)
    min_val = min(y_test_sample.min(), y_pred_sample.min())
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    ax2.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Actual Values (真實值)')
    ax2.set_ylabel('Predicted Values (預測值)')
    ax2.set_title('Actual vs. Predicted')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 自動調整佈局，避免標籤重疊
    plt.tight_layout()
    plt.savefig(f'model_evaluation_withouORIGINALPRICE_{single_target}.png', dpi=300, bbox_inches='tight')
    plt.show()


'''
    預測折舊價差
        訓練集 R2 分數: 0.24
        測試集 R2 分數: 0.11
        測試集 MAE (平均絕對誤差): 25644.73 

    預測折舊價差率
        訓練集 R2 分數: 0.88
        測試集 R2 分數: 0.85
        測試集 MAE (平均絕對誤差): 0.04

預測折舊價差分析
    少了 original_price 後，模型訓練變慢，是因為原本有 original_price 時，每一棵樹只需要生長得比較『淺』且『簡單』就能達到高純度；
    現在失去了這個強特徵，模型被迫要長出更『深』、更『複雜』的樹，才能在剩餘的特徵中挖掘足夠的資訊。

    效果也很差，表示模型無法準確預測

預測折舊價差率分析
    預測折舊價差率的分析依然表現很好，也是我們預期當中的，因為本來他就不依賴original_price

綜合分析
    為什麼兩個預測目標會有那麼龐大的差距呢，原因就是「預測目標正規化」，前者範圍從幾千到上萬，後者則是0~1的範圍
    正規化對模型訓練的影響大，是因為它將數據的「物理外表」（金額）剝離，只保留了「市場邏輯」（比例）
    這對於模型訓練有非常好的影響

下一關T3.py，我們把隨機森林拿來做分類問題
'''