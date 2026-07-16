'''
這裡是隨機森林的區塊
隨機森林是整合多個決策樹的判斷後，得到最後輸出的模型
並不是使用多種不同模型
隨機森林裡面的每一個決策樹除了會使用不同的sample(row)，也會使用不一樣的特徵(column)，達到兩層的隨機抽樣
例如
    第一個樹使用id1~50的資料，使用1,2,3,4,5號特徵
    第一個樹使用id51~100的資料，使用1,6,7,4,5號特徵
所以數據集特徵數量多會更好

使用多種模型判斷後，再經由一個特別模型去選擇要聽哪一個模型的判斷的作法叫做Stacking，訓練時間和複雜程度高了不少

隨機森林可以處理回歸和分類兩種問題，非常萬用

樹系列算法的特色：根本不需要進行標準化或處理。這些模型在尋找分割點（split point）時，只依賴數值的相對大小，而與數值的絕對大小無關。對這些模型進行標準化反而可能多此一舉。
因此，雖然有年份（大於千）、分數（0~100）、價錢（萬）等數字差異很大的特徵，依然不需要標準化


我們這次使用use_phone數據集
特徵和樣本數量都夠大

我們要透過所有的數據去推論出折舊價差

特徵有原始價格和2手價格，可以推算出折舊價差
折舊價差 = 原始價格 - 2手價格
折舊價差率 = 2手價格/原始價格

多重共線性：
    並且因為折舊價差是原始數據推論出來的衍生數據，會和原始數據有高度相關，
    為了降低複雜度和學習負擔，我們可以考慮把2手價格欄位刪除


透過隨機森林自帶的函式找出對折舊價差影響最大的特徵
model.feature_importances_




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
target = ['diff_price', 'diff_price_ratio']
X = df.drop(columns=target)
for single_target in target:
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
        max_samples=0.5,        # 每次只用 50% 的數據訓練一棵樹，顯著加速且通常能減少過擬合
        random_state=42
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
    plt.savefig(f'model_evaluation_{single_target}.png', dpi=300, bbox_inches='tight')
    plt.show()

    '''
    預測折舊價差
        預測目標: diff_price
        訓練集 R2 分數: 0.99
        測試集 R2 分數: 0.99
        測試集 MAE (平均絕對誤差): 2897.29
    預測折舊價差率
        訓練集 R2 分數: 0.89
        測試集 R2 分數: 0.86
        測試集 MAE (平均絕對誤差): 0.04

    折舊價差分析
        從數據可以看出預測能力很好
        但0.99的R2分數會讓我們懷疑他是否『作弊』
        我們可以從畫出來的圖中發現original_price的重要性遠大於其他特徵
        因為diff_price = original_price - resale_price，本質上屬於線性關係
        我們可以刪掉original_price後再來訓練一次，看效果會不會差很多

    折舊價差率分析
        效果也很好，從圖片可看出特徵之間的重要性相對平均，其中最重要的是年齡，符合我們對於折舊價格的經驗判斷
        且因為他並沒有和任何特徵有簡單的線性關係，呈現出來的重要性分布和預測圖會更符合市場現況
    
    綜合分析
        我們必須慎選預測目標，才能解讀出我們真的要的數據
        透過折舊價差率的分析，我們可以看出真的影響2手價格的因素，如年齡、電池健康度等等
        也才能從偏差值中解讀出特殊訊息，如實際折舊率比預測折舊率低的手機，也就是比預期來的保值，會不會蘋果的手機佔多數？
        您可以找幾台 Actual 遠高於 Predicted（落在對角線上方）的手機，分析它們的特徵。這些手機代表「比市場平均更保值」的產品，可能是特定的熱門品牌或型號。

        下一步，我們T2.py中測試看看預測折舊價差，但把original_price拿掉會如何
    '''