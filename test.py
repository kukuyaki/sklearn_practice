import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 1. 產生數據
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# 2. 設定視覺化網格 (建立一個背景平面)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

# 3. 建立模型比較
models = [SVC(kernel='linear'), SVC(kernel='rbf', C=5, gamma=1)]
titles = ['Linear Kernel', 'RBF Kernel']

# 4. 畫圖
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, clf in enumerate(models):
    clf.fit(X, y)
    # 預測整個網格平面
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 畫出決策邊界區域
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # 畫出原始點
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    axes[i].set_title(titles[i])

plt.show()