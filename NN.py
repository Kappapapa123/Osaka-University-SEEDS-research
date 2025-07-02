# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:05:42 2023

@author: keishi
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 入力層と隠れ層を繋ぐ(W1)と隠れ層と出力層を繋ぐ(W2)の重みの初期化
        #　np.random.randnは平均0、標準偏差1の正規分布（ガウス分布）の乱数の生成関数
        # 括弧の中身は配列の次元の設定
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
        #　隠れ層(b1）と出力層（b2）の各ニューロンの総入力の初期化
        #　np.zerosは全て値が0の指定されたサイズの配列を返す関数
        self.b1 = np.zeros(self.hidden_size)
        self.b2 = np.zeros(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # 隠れ層の計算
        hidden_layer = np.dot(X, self.W1) + self.b1
        self.hidden_output = self.sigmoid(hidden_layer)

        # 出力層の計算
        output_layer = np.dot(self.hidden_output, self.W2) + self.b2
        output_probs = self.softmax(output_layer)

        return output_probs

    def backward(self, X, y, output_probs, learning_rate):
        # 出力層の誤差と勾配の計算
        error_output = output_probs - y
        dW2 = np.dot(self.hidden_output.T, error_output)
        db2 = np.sum(error_output, axis=0)

        # 隠れ層の誤差と勾配の計算
        error_hidden = np.dot(error_output, self.W2.T) * self.sigmoid_derivative(self.hidden_output)
        dW1 = np.dot(X.T, error_hidden)
        db1 = np.sum(error_hidden, axis=0)

        # 重みとバイアスの更新
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# CSVファイルからデータを読み込む
data = pd.read_csv("Dry_Bean_Dataset_Reduced.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 入力データとクラスラベルをNumPy配列に変換
X = np.array(X)
y = np.array(y)

# 入力データとクラスラベルを正規化 (適宜スケーリング方法を選択)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# クラスラベルをone-hotエンコーディングに変換
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# ニューラルネットワークの構築
input_size = X_normalized.shape[1]
hidden_size = 20  # 隠れ層のノード数（適宜調整してください）
output_size = y_onehot.shape[1]  # クラス数に合わせる

# ニューラルネットワークのインスタンス化
nn = NeuralNetwork(input_size, hidden_size, output_size)

max_accuracy = 0
max_accuracy_time = 0
max_learning_rate = 0

# 訓練データとテストデータに分割
for j in range(40):
    total_accuracy = 0
    total_time = 0
    for i in range(10):
        X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(X_normalized, y_onehot, test_size=0.33, random_state=37+i)
        
        # バックプロパゲーションによる学習
        epochs = 150
        learning_rate = 0.0001*(j+1)
        
        # 処理の開始時間を記録
        start_time = time.time()
        
        for epoch in range(epochs):
            # 順伝播
            output_probs = nn.forward(X_train)
        
            # バックプロパゲーション
            nn.backward(X_train, y_train_onehot, output_probs, learning_rate)
        
        # 処理の終了時間を記録
        end_time = time.time()
        
        # 実行時間を計算して表示
        execution_time = end_time - start_time
        execution_time = round(execution_time, 10)
        
        # テストデータでニューラルネットワークの性能を評価
        test_predictions = nn.forward(X_test)
        
        # 正答率を計算して出力
        test_predictions_label = np.argmax(test_predictions, axis=1)
        y_test_label = np.argmax(y_test_onehot, axis=1)
        accuracy = accuracy_score(y_test_label, test_predictions_label)
        accuracy = round(accuracy, 10)
        total_accuracy += accuracy
        total_time += execution_time
        
    print(total_accuracy/10)
    if total_accuracy/10 > max_accuracy/10:
        max_accuracy = total_accuracy
        max_accuracy_time = total_time
        max_learning_rate = learning_rate

print(max_accuracy/10)
print(max_accuracy_time/10)
print(max_learning_rate)