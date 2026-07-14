#import 需要的工具
#數據處理
#sklearn前處理
#sklearn模型
#sklearn評分
#sklearn其他工具
#畫圖
#儲存model工具 import joblib

# 1. 讀取與篩選資料
# 2. 切分資料訓練與測試集
# 3. 建立 Pipeline，選擇要用的資料前處理工具
# 4. 模型優化與調參GridSearchCV vs RandomizedSearchCV
# 4. 訓練資料，fit
# 5. 評估結果
# 6. 實際預測範例


#實用程式碼範例AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA25
#1
'''
將數據分成數字（模型可以直接理解）或文字特徵（需要轉換）
數字：標準化
文字：one hot encoder編碼
'''
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 在 Pipeline 中分別處理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),           # 數值型做標準化
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # 類別型做編碼
    ])



#2
'''
建立 Pipeline
Pipeline(steps=[
    ('步驟名稱1', 轉換器物件),
    ('步驟名稱2', 轉換器物件),
    ('最後的工站', 模型物件)
])
轉換器物件要可以有要有 fit 和 transform 方法
估計器Estimator, 要有 fit 方法（通常就是你的模型
'''
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



#3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#4
y = df['goals']
print(y.value_counts())


#5
scaler = StandardScaler()
regressor = LinearRegression(n_jobs=-1)

X_train_scaled = scaler.fit_transform(X_train)
regressor.fit(X_train_scaled, y_train)

#上面等於下面
model = Pipeline(steps=[
    ('preprocessor', StandardScaler()
        ),
    ('classifier', LinearRegression(
        n_jobs=-1              # 使用所有 CPU 核心加速運算
    ))
])

model.fit(X_train, y_train)



#6
from sklearn import metrics

# --- 分類指標 ---
# y_true: 真實標籤, y_pred: 模型預測標籤
print(metrics.accuracy_score(y_true, y_pred))
print(metrics.classification_report(y_true, y_pred)) # 一次顯示 P, R, F1

# --- 迴歸指標 ---
# y_true: 真實數值, y_pred: 模型預測數值
print(metrics.r2_score(y_true, y_pred))
print(metrics.mean_absolute_error(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred, squared=False)) # RMSE
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
    '''




#7
'''
各個特徵的相關性，使用這個程式碼可幫助挑選預測輸出特徵時要用的輸入特徵
'''
numeric_df = df.select_dtypes(include=[np.number])

# 2. 對純數字的 DataFrame 計算相關係數並畫圖
plt.figure(figsize=(12, 10)) # 建議加上這行，避免圖太擠
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
