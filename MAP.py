'''
第一階段：初階 — 建立穩固的建模基礎 (建立「工具箱」)
這個階段的目標是讓你從「會調用函式庫」進化到「理解模型內部的運作」。

資料預處理與清理：

Pandas 進階操作（groupby, pivot, apply）。

數據缺失值處理（Mean/Median Imputation, KNN Imputation）。

特徵工程：標準化 (Standardization)、歸一化 (Normalization)、類別變數編碼 (One-Hot Encoding)。

監督式學習 (基本模型)：

線性模型：線性迴歸 (Linear Regression)、邏輯迴歸 (Logistic Regression)。

樹模型基礎：決策樹 (Decision Trees)、隨機森林 (Random Forest)。

模型評估與調參：

訓練集與測試集劃分 (Train-Test Split)、K-Fold 交叉驗證。

指標：Accuracy, Precision, Recall, F1-Score, RMSE, MAE。

調參：Grid Search, Random Search。
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
第二階段：中階 — 處理複雜場景與效率優化 (成為「工程師」)
這個階段的重點是解決真實世界中「髒數據」與「高難度任務」的問題。

梯度提升樹 (GBDT) 體系：

深入理解 XGBoost, LightGBM, CatBoost 的運作機制。

比較它們在你的足球數據上的表現差異與訓練效率。

非監督式與特徵處理：

降維：PCA (主成分分析)、t-SNE (用於視覺化)。

聚類：K-Means, DBSCAN。

進階特徵工程：

處理共線性 (Regularization: Lasso, Ridge)。

處理不平衡數據 (SMOTE, Class Weight Adjustment)。

機器學習管線 (Pipeline)：

學習使用 scikit-learn Pipeline 將預處理、特徵工程與建模流程封裝，確保預測流程的嚴謹性（避免資料洩漏）。
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
第三階段：高階 — 深度探索與系統架構 (成為「架構師」)
這個階段是為了處理大規模數據、複雜結構數據，以及實現自動化流程。

深度學習 (Deep Learning)：

多層感知器 (MLP) 用於複雜非線性預測。

Pytorch 框架基礎（這是目前學術與研究的首選）。

時間序列分析：

對於球隊財務、球員長期狀態，學習 ARIMA, LSTM, 或 Prophet 模型。

自然語言處理 (NLP) 應用：

文字向量化 (TF-IDF, Word2Vec, BERT Embeddings)。

將球探報告或新聞數據納入你的足球預測模型中。

進階整合與部署：

Stacking / Blending：模型堆疊策略，追求競賽級的精確度。

MLOps 觀念：學習如何將模型部署為 API (FastAPI) 並監控模型漂移 (Model Drift)。

解釋性人工智慧 (XAI)：使用 SHAP 或 LIME 來解釋黑盒子模型，向教練或決策者解釋預測背後的關鍵因子。
'''