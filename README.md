# 機器學習


機器學習 (Machine Learning)
│
├── 1. 監督學習 (Supervised Learning)
│   ├── 傳統演算法 (Classical Algorithms)
│   │   ├── 線性模型: 線性迴歸, Logistic 迴歸, Ridge/Lasso
│   │   ├── 決策樹系列: Decision Trees, Random Forest, GBDT, XGBoost, LightGBM
│   │   └── 向量空間模型: SVM, Naive Bayes, k-NN
│   └── 監督式深度學習 (Deep Learning)
│       ├── MLP (多層感知器 / 全連接網路)
│       ├── CNN (卷積神經網路): ResNet, EfficientNet, MobileNet
│       └── Transformer (監督式): BERT, RoBERTa (分類/標註任務)
│
├── 2. 非監督學習 (Unsupervised Learning)
│   ├── 分群演算法 (Clustering)
│   │   ├── 基於距離: k-Means, DBSCAN, Hierarchical Clustering
│   │   └── 基於機率: Gaussian Mixture Models (GMM)
│   ├── 降維演算法 (Dimensionality Reduction)
│   │   ├── 線性: PCA (主成分分析), LDA
│   │   └── 非線性: t-SNE, UMAP, Kernel PCA
│   └── 非監督式深度學習 (Deep Learning)
│       ├── Autoencoders (AE, VAE, Sparse AE)
│       ├── GAN (生成對抗網路): DCGAN, StyleGAN
│       └── Self-Supervised (自監督): GPT (預測下一個 Token), SimCLR
│
└── 3. 強化學習 (Reinforcement Learning)
    ├── 基於價值 (Value-based)
    │   ├── Q-Learning, SARSA
    │   └── DQN (Deep Q-Network)
    ├── 基於策略 (Policy-based)
    │   ├── Policy Gradient (REINFORCE)
    │   └── PPO (Proximal Policy Optimization)
    └── 演員-評論家 (Actor-Critic)
        ├── A3C, DDPG, SAC (Soft Actor-Critic)
        └── Decision Transformer (將 RL 視為序列建模)

非監督式學習是「嬰兒的自主觀察」，是直覺，讓機器能透過觀察發現規律，並寫下規律，如書本
監督式學習是「老師指導的課堂」，是智慧，讓機器閱讀書本資料後變得聰明，能強化自身的判斷
強化學習是「生存的競技場」，是肌肉，將學習來的強大判斷能力，變成和現實世界互動的可能

非監督學習（混沌中的秩序）：AI 觀察世界，理解特徵分布、降維、歸納規律。——「建構世界觀」
監督學習（認知的定義）：AI 藉由人類標註的知識，學會對特定的規律進行定義和預測。——「建立知識庫」
強化學習（行動的智慧）：AI 基於前兩者提供的「世界觀」與「知識」，透過試錯來優化行動策略，達成特定目標。——「實現決策力」

非監督學習是底層的「語感與直覺」。
監督學習是中層的「知識與分類」。
強化學習是頂層的「策略與行動」。


非監督學習
    「這些東西為什麼會湊在一起？」
    「哪裡跟哪裡不一樣？」、
    「有沒有什麼關鍵特徵能描述這群東西？」
監督學習
    「這是 A 還是 B？」
    「這條線的預測值是多少？」
強化學習
    「為了達到目的，我現在該採取什麼行動？」

    
非監督學習：是在繪製地圖。它先觀察地形，告訴你哪裡是高山（群集）、哪裡是平原，幫你把世界分成幾個不同的區域。
監督學習：是在地圖上標示地標。它在地圖的基礎上，幫你貼上標籤：「這座山叫玉山」、「那片平原是台北」。
強化學習：是在規劃路徑。它看著標示好的地圖，尋找從起點到終點最快、最省力、最安全的走法。