'''
之後要來寫unsupervised的區塊
以下是常見的unsupervised的模型
叢集    
    K-Means 
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    階層式分群 (Hierarchical Clustering / Agglomerative)
    高斯混合模型 (Gaussian Mixture Models, GMM)
    譜分群 (Spectral Clustering)

關聯規則學習
    Apriori 演算法
    Eclat 演算法 (Equivalence Class Transformation)
    FP-Growth (Frequent Pattern Growth)
    AssocRule (基於決策樹的關聯挖掘)
    序列模式挖掘 (Sequence Pattern Mining / GSP)
概率密度
    核密度估計 (Kernel Density Estimation, KDE)
    高斯核密度估計 (Gaussian KDE)
    變分自動編碼器 (Variational Autoencoders, VAE)
    生成對抗網路 (Generative Adversarial Networks, GANs)
    Parzen Window 方法
尺寸減少
    主成分分析 (Principal Component Analysis, PCA)
    t-分散鄰域嵌入 (t-SNE)
    UMAP (Uniform Manifold Approximation and Projection)
    線性判別分析 (Linear Discriminant Analysis, LDA)
    自動編碼器 (Autoencoder)


我們已經用過PCA了
這次來使用BDSCAN，他相較於k means，在對於非圓形的叢集能更好的分群
叢集算法在多特徵的數據集表現會下降，因此我們還是可以用PCA先降低維度

我們這次要使用的數據集是遊戲成癮數據集
基本資料
    user_id
    age
    gender
    country
    occupation
    income_level
遊戲習慣
    years_gaming
    preferred_genre
    platform
    device_type
    rank_tier
    daily_playtime_hours
    weekly_play_sessions
    late_night_sessions_hours
    weekend_playtime_hours
    consecutive_hours_max
    multiplayer_ratio
遊戲內行為與消費
    toxic_chat_reports
    rage_quit_frequency
    in_game_purchases
    monthly_spending_usd
    lootbox_openings
    subscription_status
心理與健康指標
    stress_score
    loneliness_score
    dopamine_dependency_index
    self_control_score
    impulsiveness_score
    anxiety_level
    depression_indicator
    emotional_stability
    sleep_hours
    exercise_frequency_per_week
    caffeine_intake_cups_day
生活影響與表現
    social_interaction_hours
    relationship_status
    gpa_or_performance_score
    missed_deadlines
    productivity_drop_percent
    absenteeism_days
    internet_speed_mbps
    screen_time_total_hours
分析變數
    behavioral_cluster
    addiction_score
    addiction_binary
    addiction_severity
    burnout_probability
    mental_health_risk_score
    churn_probability
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.impute import SimpleImputer

# 1. 載入資料
df = pd.read_csv('/home/kgforsure/Documents/code/sklearn_practice/gameData/gaming_addiction.csv')

# 針對數值欄位填充平均值，針對類別欄位填充眾數 (最常出現的值)
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

# 數值補平均值
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# 類別補眾數
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])



# 2. 將字串數字化
# 識別所有類別型特徵 (object type)
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

# 使用 LabelEncoder 進行簡單轉換 (針對無序類別，One-Hot 亦可，視特徵基數而定)
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# 3. 標準化 (PCA 與 DBSCAN 極度依賴標準化)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)




# 4. PCA 分析 (降維至 5 個成分)
pca = PCA(n_components=5)
pca_result = pca.fit_transform(df_scaled)

# 查看解釋變異數比例，確認 5 個成分是否足夠
print(f"Explained variance ratio of 5 components: {pca.explained_variance_ratio_.sum():.2f}")
print(f"importance of 5 components: {pca.explained_variance_ratio_}")
# 1. 建立一個包含權重的 DataFrame
# pca.components_ 的形狀是 [n_components, n_features]
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
    index=df_encoded.columns # 確保你的 df_encoded 是 PCA 輸入前的那個 DataFrame
)
# 2. 顯示每個主成分中「影響力最大」的前幾個特徵
for i in range(pca.n_components_):
    component_name = f'PC{i+1}'
    # 取出該成分的權重，並按絕對值排序 (因為負數代表反向影響)
    top_features = loadings[component_name].reindex(
        loadings[component_name].abs().sort_values(ascending=False).index
    )
    print(f"\n--- {component_name} 的主要特徵貢獻 ---")
    print(top_features.head(10)) # 印出前 10 個影響最大的特徵




# 5. DBSCAN 分析
# eps: 鄰域半徑, min_samples: 成為核心點的最小樣本數
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(pca_result)
df['cluster'] = labels





# 6. 繪圖
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
pca_df['cluster'] = labels  # 加入分群標籤

# 第一張圖：使用 pairplot 繪製所有主成分的兩兩對比矩陣
# 這會自動生成 5x5 的矩陣圖，對角線為分佈圖，非對角線為散點圖
sns.pairplot(pca_df, hue='cluster', palette='viridis', diag_kind='kde', markers='o')
plt.suptitle('Pairwise Comparison of PCA Components by Cluster', y=1.02)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 準備資料：選擇你想要觀察的特徵
features_to_plot = [
    'age', 
    'daily_playtime_hours', 
    'mental_health_risk_score', 
    'monthly_spending_usd', 
    'self_control_score'
]

# 2. 將 DataFrame 轉換為長格式 (Long format) 以便 seaborn 繪圖
# 這會將選定的欄位堆疊，形成一個適合畫圖的結構
df_melted = df.melt(id_vars=['cluster'], value_vars=features_to_plot, var_name='Feature', value_name='Value')

# 3. 建立一個 3x2 的子圖網格 (一共 5 個變數，所以會有一個空間留白)
fig, axes = plt.subplots(3, 2,figsize=(12, 10), dpi=100,constrained_layout=True)
axes = axes.flatten() # 方便迭代

# 4. 迴圈繪圖
for i, feature in enumerate(features_to_plot):
    sns.boxplot(x='cluster', y='Value', data=df_melted[df_melted['Feature'] == feature], ax=axes[i])
    axes[i].set_title(f'{feature} by Cluster')
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

# 隱藏多餘的網格
axes[5].axis('off')

plt.tight_layout()
plt.show()

'''
先處理數據
然後pca分成5個主成分
跑BDSCAN
繪圖

主要成分
    PCA1:daily_playtime_hours 
    PCA2:mental_health_risk_score 
    PCA3:monthly_spending_usd  
    PCA4:age   
    PCA5:self_control_score   

從圖形一我們可以看出，pca1,4,5有明顯分別
其中pca1代表成癮性
其中pca4代表經濟能力
其中pca5代表情緒表現

pca2代表心理健康
pca3代表消費情形



觀察圖形一和二後，可能的結論
    經濟能力表現好的人群，遊玩時間也比較長
    遊戲時長和心理健康可能是0相關
    

    高強度遊戲群 (Cluster 0)：以長時間的高專注度投入為主，行為模式高度一致且穩定，但屬於心理健康風險較高的一群。
    消費型玩家群 (Cluster 1)：遊戲時間適中，但顯示出較高的衝動消費傾向與較低的自制力得分，可能將經濟投入作為心理代償的手段。
    健康娛樂型玩家 (Cluster 2)：具備最高的自我控制力，遊戲時間短且消費極低，代表了一種健康的休閒模式。



技術解析
pca主成分分析要特別觀察的幾個數據
1.主成分數量                  PCA(n_components=5)
2.各個主成分的各項特徵重要性     pca.components_
3.所有主成分的資訊轉化率        pca.explained_variance_ratio_.sum

BDSCAN重要參數
1.eps : 在多維度數據集當中，密度會趨於平均，導致能以劃分不同群，且因為維度變大，導致eps要設定的更大才能抓取到相鄰點
2.min_sample ： 最少要多少點才能作為核心點
DBSCAN(eps=1.5, min_samples=5)

這些參數要調整到剛剛好才能有好的圖形


非監督學習沒有下一關了
因為我要體醒讀者
你可能發現，跑這段程式碼時間是很快的
因為他本質只是在計算距離

其他非監督學習，如關聯規則學習的技術，可能只是統計學的計算而已
簡單的計算可以解決的事情，不用特地去使用複雜的模型
非監督學習接下來的學習目標，將不會是要套用什麼模型，而是如何去解讀數據
"奧卡姆剃刀"

當然，除非複雜度超高，可能要用到深度學習的情況，我們再來探討模型的應用




下一步我們先研究簡單的強化學習Q learning，把學習範式三巨頭都走過一遭

要注意，強化學習 相對於我們學到的 監督與非監督 來說 難度高上不少
你能走到這裡，已經是非常了不起了，要是你還能通過強化學習的練習，你要為自己感到驕傲

等學完後，你可以驕傲的說你已經正式通過新手教學
即將進入專家領域，學習深度學習，應用深度學習，正式進入AI發展的大時代

並且開始思考你要做的研究和方向，從 單純的學習者 變成 學習者and給予者（你依然還是得學習）

專家領域                                前沿領域
    從更難的強化學習，如DQN                    語言模型參數微調
    深度學習的3D空間生成                       實體機器人 Sim-to-Real
    機器人導航與探索                           因果發現與動態干預
    多代理人強化學習                           領域專用 AI 基礎架構
    機器學習系統設計                           等等
    等等

下一關，請到另一個repositories，RL_prtice，進入強化學習第一課

補充：特地分成不同的repositories存放的原因
    1.sklearn專注在監督與非監督學習上，而強化學習將使用不同的函式庫
    2.也因為強化學習是另一個難度的東西，即使三者都屬於機器學習三大範式，我依然打算將他與監督與非監督分開
    3.深度學習也可能會自成一repositories
    4.之後就是各個專題各自的repositories了，希望這樣不會太複雜
'''