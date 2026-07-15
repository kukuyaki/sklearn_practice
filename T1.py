'''
這裡是隨機森林的區塊
隨機森林是整合多個決策樹的判斷後，得到最後輸出的模型
並不是使用多種不同模型

使用多種模型判斷後，再經由一個特別模型去選擇要聽哪一個模型的判斷的作法叫做Stacking，訓練時間和複雜程度高了不少

隨機森林可以處理回歸和分類兩種問題，非常萬用
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns