# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 00:20:33 2023

@author: keishi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

# CSVファイルからデータを読み込む
data = pd.read_csv("Dry_Bean_Dataset.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 入力データとクラスラベルをNumPy配列に変換
X = np.array(X)
y = np.array(y)

# クラスラベルを数値に変換
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 入力データとクラスラベルを正規化 )
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

total_accuracy = 0
total_time = 0

for i in range(10):
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split\
        (X_normalized, y, test_size=0.33, random_state=37+i)
    
    # 処理の開始時間を記録
    start_time = time.time()
    
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    
    # 処理の終了時間を記録
    end_time = time.time()
    
    # 実行時間を計算して表示
    execution_time = end_time - start_time
    
    # サポートベクターマシンの予測
    test_predictions = svm.predict(X_test)
    
    # 正答率を計算して出力
    accuracy = accuracy_score(y_test, test_predictions)
    execution_time = round(execution_time, 8)
    accuracy = round(accuracy, 8)
    total_accuracy += accuracy
    total_time += execution_time
    print(accuracy)
    print(execution_time)
    
print(total_accuracy/10)
print(total_time/10)

