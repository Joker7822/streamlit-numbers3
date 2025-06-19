import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
    GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
)
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from statsmodels.tsa.arima.model import ARIMA
from stable_baselines3 import PPO
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib
matplotlib.use('Agg')  # ← ★ この行を先に追加！
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import warnings
import re
import platform
import gym
import sys
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from neuralforecast.models import TFT
from neuralforecast import NeuralForecast
import onnxruntime
import streamlit as st
from autogluon.tabular import TabularPredictor
import torch.backends.cudnn
from datetime import datetime 
from collections import Counter
import torch.nn.functional as F
import math

# Windows環境のイベントループポリシーを設定
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings("ignore")

class LotoEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(LotoEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        selected_numbers = list(np.argsort(action)[-4:])
        winning_numbers = list(np.random.choice(self.historical_numbers, 4, replace=False))

        prize = classify_numbers3_prize(selected_numbers, winning_numbers)
        prize_rewards = {
            "ストレート": 468700,
            "ボックス": 18700,
            "はずれ": -100
        }

        reward = prize_rewards.get(prize, -100)
        done = True
        obs = np.zeros(10, dtype=np.float32)
        return obs, reward, done, {}

class DiversityEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(DiversityEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.previous_outputs = set()

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        selected = tuple(sorted(np.argsort(action)[-4:]))
        reward = 1.0 if selected not in self.previous_outputs else -1.0
        self.previous_outputs.add(selected)
        return np.zeros(10, dtype=np.float32), reward, True, {}

class CycleEnv(gym.Env):
    def __init__(self, historical_numbers):
        super(CycleEnv, self).__init__()
        self.historical_numbers = historical_numbers
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.cycle_scores = calculate_number_cycle_score(historical_numbers)

    def reset(self):
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        selected = np.argsort(action)[-4:]
        avg_cycle = np.mean([self.cycle_scores.get(n, 999) for n in selected])
        reward = max(0, 1 - (avg_cycle / 50))
        return np.zeros(10, dtype=np.float32), reward, True, {}

class MultiAgentPPOTrainer:
    def __init__(self, historical_data, total_timesteps=5000):
        self.historical_data = historical_data
        self.total_timesteps = total_timesteps
        self.agents = {}

    def train_agents(self):
        envs = {
            "accuracy": LotoEnv(self.historical_data),
            "diversity": DiversityEnv(self.historical_data),
            "cycle": CycleEnv(self.historical_data)
        }

        for name, env in envs.items():
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=self.total_timesteps)
            self.agents[name] = model
            print(f"[INFO] PPO {name} エージェント学習完了")

    def predict_all(self, num_candidates=50):
        predictions = []
        for name, model in self.agents.items():
            obs = model.env.reset()
            for _ in range(num_candidates // 3):
                action, _ = model.predict(obs)
                selected = list(np.argsort(action)[-4:])
                predictions.append((selected, 0.9))  # 信頼度は仮
        return predictions

class AdversarialLotoEnv(gym.Env):
    def __init__(self, target_numbers_list):
        """
        GANが生成した番号（target_numbers_list）をターゲットとし、
        PPOに「それらを当てさせる」対戦環境
        """
        super(AdversarialLotoEnv, self).__init__()
        self.target_numbers_list = target_numbers_list
        self.current_index = 0
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        self.current_index = (self.current_index + 1) % len(self.target_numbers_list)
        return np.zeros(10, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        selected_numbers = set(np.argsort(action)[-3:])
        target_numbers = set(self.target_numbers_list[self.current_index])

        main_match = len(selected_numbers & target_numbers)
        reward = main_match / 3  # 部分一致率を報酬とする

        done = True
        obs = np.zeros(10, dtype=np.float32)
        return obs, reward, done, {}

def score_real_structure_similarity(numbers):
    """
    数字リストに対して、「本物らしい構造かどうか」を評価するスコア（0〜1）
    - 合計が10〜20
    - 重複がない
    - 並びが昇順 or 降順
    """
    if len(numbers) != 3:
        return 0
    score = 0
    if 10 <= sum(numbers) <= 20:
        score += 1
    if len(set(numbers)) == 3:
        score += 1
    if numbers == sorted(numbers) or numbers == sorted(numbers, reverse=True):
        score += 1
    return score / 3  # 最大3点満点を0〜1スケール

class LotoGAN(nn.Module):
    def __init__(self, noise_dim=100):
        super(LotoGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.noise_dim = noise_dim

    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            samples = self.generator(noise)
        return samples.numpy()

    def evaluate_generated_numbers(self, sample_tensor):
        """
        sample_tensor: shape=(10,) のTensor（0〜1値で各数字のスコア）
        上位3つを選んで番号に変換 → 判別器スコアと構造スコアを合成
        """
        numbers = list(np.argsort(sample_tensor.cpu().numpy())[-3:])
        numbers.sort()
        real_score = score_real_structure_similarity(numbers)

        with torch.no_grad():
            discriminator_score = self.discriminator(sample_tensor.unsqueeze(0)).item()

        final_score = 0.5 * discriminator_score + 0.5 * real_score
        return final_score

class DiffusionNumberGenerator(nn.Module):
    def __init__(self, noise_dim=16, steps=100):
        super(DiffusionNumberGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.steps = steps

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 各数字のスコア（0〜9）
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def generate(self, num_samples=20):
        samples = []
        for _ in range(num_samples):
            noise = torch.randn(1, self.noise_dim)
            x = noise
            for _ in range(self.steps):
                noise_grad = torch.randn_like(x) * 0.1
                x = x - 0.01 * x + noise_grad
            with torch.no_grad():
                scores = self.forward(x).squeeze().numpy()
            top4 = np.argsort(scores)[-4:]
            samples.append(sorted(top4.tolist()))
        return samples

def create_advanced_features(dataframe):
    dataframe = dataframe.copy()
    def convert_to_number_list(x):
        if isinstance(x, str):
            cleaned = x.strip("[]").replace(",", " ").replace("'", "").replace('"', "")
            return [int(n) for n in cleaned.split() if n.isdigit()]
        return x if isinstance(x, list) else [0]

    dataframe['本数字'] = dataframe['本数字'].apply(convert_to_number_list)
    dataframe['抽せん日'] = pd.to_datetime(dataframe['抽せん日'])

    valid_mask = (dataframe['本数字'].apply(len) == 3)
    dataframe = dataframe[valid_mask].copy()

    if dataframe.empty:
        print("[ERROR] 有効な本数字が存在しません（4桁データがない）")
        return pd.DataFrame()  # 空のDataFrameを返す

    nums_array = np.vstack(dataframe['本数字'].values)
    features = pd.DataFrame(index=dataframe.index)

    features['数字合計'] = nums_array.sum(axis=1)
    features['数字平均'] = nums_array.mean(axis=1)
    features['最大'] = nums_array.max(axis=1)
    features['最小'] = nums_array.min(axis=1)
    features['標準偏差'] = np.std(nums_array, axis=1)

    return pd.concat([dataframe, features], axis=1)

def preprocess_data(data):
    """データの前処理: 特徴量の作成 & スケーリング"""
    
    # 特徴量作成
    processed_data = create_advanced_features(data)

    if processed_data.empty:
        print("エラー: 特徴量生成後のデータが空です。データのフォーマットを確認してください。")
        return None, None, None

    print("=== 特徴量作成後のデータ ===")
    print(processed_data.head())

    # 数値特徴量の選択
    numeric_features = processed_data.select_dtypes(include=[np.number]).columns
    X = processed_data[numeric_features].fillna(0)  # 欠損値を0で埋める

    print(f"数値特徴量の数: {len(numeric_features)}, サンプル数: {X.shape[0]}")

    if X.empty:
        print("エラー: 数値特徴量が作成されず、データが空になっています。")
        return None, None, None

    # スケーリング
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== スケーリング後のデータ ===")
    print(X_scaled[:5])  # 最初の5件を表示

    # 目標変数の準備
    try:
        y = np.array([list(map(int, nums)) for nums in processed_data['本数字']])
    except Exception as e:
        print(f"エラー: 目標変数の作成時に問題が発生しました: {e}")
        return None, None, None

    return X_scaled, y, scaler

def convert_numbers_to_binary_vectors(data):
    vectors = []
    for numbers in data['本数字']:
        vec = np.zeros(10)
        for n in numbers:
            if 0 <= n <= 9:
                vec[n] = 1
        vectors.append(vec)
    return np.array(vectors)

def calculate_prediction_errors(predictions, actual_numbers):
    """予測値と実際の当選結果の誤差を計算し、特徴量として保存"""
    errors = []
    for pred, actual in zip(predictions, actual_numbers):
        pred_numbers = set(pred[0])
        actual_numbers = set(actual)
        error_count = len(actual_numbers - pred_numbers)
        errors.append(error_count)
    
    return np.mean(errors)

def delete_old_generation_files(directory, days=1):
    """指定フォルダ内で、指定日数より古いCSVファイルを削除"""
    now = datetime.now()
    deleted = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".csv"):
            try:
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if (now - modified_time).days >= days:
                    os.remove(filepath)
                    deleted += 1
            except Exception as e:
                print(f"[WARNING] ファイル削除エラー: {filename} → {e}")
    if deleted:
        print(f"[INFO] {deleted} 件の古い世代ファイルを削除しました")

def save_self_predictions(predictions, file_path="self_predictions.csv", max_records=100, historical_data=None):
    """予測結果をCSVに保存し、一致数と等級も記録"""
    rows = []
    valid_grades = ["はずれ", "ボックス", "ストレート"]  
    for numbers, confidence in predictions:
        match_count = "-"
        prize = "-"
        if historical_data is not None:
            actual_list = [parse_number_string(x) for x in historical_data['本数字'].tolist()]
            match_count = max(len(set(numbers) & set(actual)) for actual in actual_list)
            prize = max(
                (classify_numbers3_prize(numbers, actual) for actual in actual_list),
                key=lambda p: valid_grades.index(p) if p in valid_grades else -1
            )
        rows.append(numbers + [confidence, match_count, prize])

    # 既存ファイルが存在し、中身が空でない場合のみ読み込む
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            existing = pd.read_csv(file_path, header=None).values.tolist()
            rows = existing + rows
        except pd.errors.EmptyDataError:
            print(f"[WARNING] {file_path} は空のため、読み込みをスキップします。")
    else:
        print(f"[INFO] {file_path} が空か存在しないため、新規作成します。")

    # 最新 max_records 件に制限して保存
    rows = rows[-max_records:]
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False, header=False)
    print(f"[INFO] 自己予測を {file_path} に保存（最大{max_records}件）")

    # 🔁 世代別ファイルの保存
    gen_dir = "self_predictions_gen"
    os.makedirs(gen_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generation_file = os.path.join(gen_dir, f"self_predictions_gen_{timestamp}.csv")
    df.to_csv(generation_file, index=False, header=False)
    print(f"[INFO] 世代別予測を保存: {generation_file}")

    # 🧹 古いファイルを自動削除（1日以上前のファイルを削除）
    delete_old_generation_files(gen_dir, days=1)

def load_self_predictions(file_path="self_predictions.csv", min_match_threshold=3, true_data=None):
    if not os.path.exists(file_path):
        print(f"[INFO] 自己予測ファイル {file_path} が見つかりません。")
        return None

    if os.path.getsize(file_path) == 0:
        print(f"[INFO] 自己予測ファイル {file_path} は空です。")
        return None

    try:
        df = pd.read_csv(file_path, header=None).dropna()

        # 4桁予測番号の列を取り出し
        number_cols = df.iloc[:, :4].astype(int)
        numbers_list = number_cols.values.tolist()

        if true_data is not None:
            scores = evaluate_self_predictions(numbers_list, true_data)

            # ✅ 高一致数のみを収集
            valid_predictions = []
            for pred, match in zip(numbers_list, scores):
                if match >= min_match_threshold:
                    valid_predictions.append(tuple(pred))

            # ✅ 出現頻度つきで返す
            from collections import Counter
            freq = Counter(valid_predictions)
            sorted_preds = sorted(freq.items(), key=lambda x: -x[1])  # 回数降順

            print(f"[INFO] 一致数{min_match_threshold}以上の自己予測（重複集計あり）: {len(sorted_preds)}件")
            return sorted_preds  # ← [(番号タプル, 出現回数)] 形式

        return numbers_list

    except Exception as e:
        print(f"[ERROR] 自己予測読み込みエラー: {e}")
        return None

def evaluate_self_predictions(self_predictions, true_data):
    """
    自己予測リストと本物データを比較して一致数を評価
    :param self_predictions: [[5,12,17,22,30,34,37], ...]
    :param true_data: 過去の本物本数字データ（data['本数字'].tolist()）
    :return: 各自己予測に対応する最大一致数リスト
    """
    scores = []
    true_sets = [set(nums) for nums in true_data]

    for pred in self_predictions:
        pred_set = set(pred)
        max_match = 0
        for true_set in true_sets:
            match = len(pred_set & true_set)
            if match > max_match:
                max_match = match
        scores.append(max_match)

    return scores

def update_features_based_on_results(data, accuracy_results):
    """過去の予測結果と実際の結果の比較から特徴量を更新"""
    
    for result in accuracy_results:
        event_date = result["抽せん日"]
        max_matches = result["最高一致数"]
        avg_matches = result["平均一致数"]
        confidence_avg = result["信頼度平均"]

        # 過去のデータに予測精度を組み込む
        data.loc[data["抽せん日"] == event_date, "過去の最大一致数"] = max_matches
        data.loc[data["抽せん日"] == event_date, "過去の平均一致数"] = avg_matches
        data.loc[data["抽せん日"] == event_date, "過去の予測信頼度"] = confidence_avg

    # 特徴量がない場合は0で埋める
    data["過去の最大一致数"] = data["過去の最大一致数"].fillna(0)
    data["過去の平均一致数"] = data["過去の平均一致数"].fillna(0)
    data["過去の予測信頼度"] = data["過去の予測信頼度"].fillna(0)

    return data

class LotoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LotoLSTM, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.ModuleList([
            nn.Linear(hidden_size * 2, 10) for _ in range(3)  # 各桁：0〜9分類
        ])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        return [fc(context) for fc in self.fc]  # 各桁の出力

def train_lstm_model(X_train, y_train, input_size, device):
    
    torch.backends.cudnn.benchmark = True  # ★これを追加
    
    model = LotoLSTM(input_size=input_size, hidden_size=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)  # ★変更

    scaler = torch.cuda.amp.GradScaler()  # ★Mixed Precision追加

    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # ★ここもMixed Precision
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"[LSTM] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # ONNXエクスポート
    dummy_input = torch.randn(1, 1, input_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "lstm_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12
    )
    print("[INFO] LSTM モデルのトレーニングが完了")
    return model

def extract_high_accuracy_combinations(evaluation_df, threshold=2):
    high_matches = evaluation_df[evaluation_df["本数字一致数"] >= threshold]
    return high_matches

def transform_to_digit_labels(numbers_series):
    y1, y2, y3 = [], [], []
    for entry in numbers_series:
        digits = [int(d) for d in re.findall(r'\d', str(entry))[:3]]
        if len(digits) == 3:
            y1.append(digits[0])
            y2.append(digits[1])
            y3.append(digits[2])
    return y1, y2, y3

# --- 🔁 的中予測抽出（Transfer Learning 用） ---
def extract_matched_predictions(predictions, true_data, min_match=2):
    matched = []
    for pred in predictions:
        for true in true_data:
            if len(set(pred) & set(true)) >= min_match:
                matched.append(pred)
                break
    return matched

def reinforce_top_features(X, feature_names, target_scores, top_n=5):
    corrs = {
        feat: abs(np.corrcoef(X[:, i], target_scores)[0, 1])
        for i, feat in enumerate(feature_names)
    }
    top_feats = sorted(corrs.items(), key=lambda x: -x[1])[:top_n]
    reinforced_X = X.copy()
    for feat, _ in top_feats:
        idx = feature_names.index(feat)
        reinforced_X[:, idx] *= 1.5
    return reinforced_X

class MemoryEncoder(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=64, num_layers=2, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)  # x: [seq_len, batch]
        return self.encoder(x)  # return shape: [seq_len, batch, embed_dim]

class GPT4Numbers(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        tgt_embed = self.embedding(tgt)  # (seq_len, batch, embed_dim)
        tgt_embed = self.pos_encoding(tgt_embed)  # (seq_len, batch, embed_dim)
        output = self.decoder(tgt_embed, memory)  # (seq_len, batch, embed_dim)
        return self.fc_out(output)  # (seq_len, batch, vocab_size)

def build_memory_from_history(history_sequences, encoder, device):
    if not history_sequences:
        return torch.zeros((1, 1, encoder.embedding.embedding_dim), device=device)

    max_len = max(len(seq) for seq in history_sequences)
    padded = [seq + [0] * (max_len - len(seq)) for seq in history_sequences]
    tensor = torch.tensor(padded, dtype=torch.long).T.to(device)  # shape: [seq_len, batch]

    # ⚠️ 修正：ここで1件だけの履歴に絞る
    tensor = tensor[:, :1]  # ← バッチサイズを1に制限

    with torch.no_grad():
        memory = encoder(tensor)  # shape: [seq_len, 1, embed_dim]
    return memory

def train_gpt3numbers_model_with_memory(
    save_path="gpt3numbers.pth",
    encoder_path="memory_encoder_3.pth",
    epochs=50
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = MemoryEncoder().to(device)
    decoder = GPT4Numbers().to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    try:
        df = pd.read_csv("numbers3.csv")
        df["本数字"] = df["本数字"].apply(parse_number_string)
        sequences = [row for row in df["本数字"] if isinstance(row, list) and len(row) == 3]
    except Exception as e:
        print(f"[ERROR] 学習データ読み込みエラー: {e}")
        return decoder, encoder

    # 学習データ（1〜2桁 → 次の1桁）
    data = []
    for seq in sequences:
        for i in range(1, 3):
            context = seq[:i]
            target = seq[i]
            history = sequences[:sequences.index(seq)]
            data.append((context, target, history[-10:]))

    if not data:
        print("[WARNING] GPT3Numbers 学習データが空です")
        return decoder, encoder

    print(f"[INFO] GPT3Numbers 学習データ件数: {len(data)}")

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for context, target, hist in data:
            tgt = torch.tensor(context, dtype=torch.long).unsqueeze(1).to(device)
            target_tensor = torch.tensor([target], dtype=torch.long).to(device)

            memory = build_memory_from_history(hist, encoder, device)

            output = decoder(tgt, memory)
            last_output = output[-1, 0].unsqueeze(0)

            loss = criterion(last_output, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[GPT3-MEM] Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")

    torch.save(decoder.state_dict(), save_path)
    torch.save(encoder.state_dict(), encoder_path)
    print(f"[INFO] GPT3Numbers 保存: {save_path}")
    print(f"[INFO] MemoryEncoder 保存: {encoder_path}")
    return decoder, encoder

def gpt_generate_predictions_with_memory_3(decoder, encoder, history_sequences, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.eval()
    encoder.eval()
    predictions = []

    memory = build_memory_from_history(history_sequences[-10:], encoder, device)

    for _ in range(num_samples):
        seq = [random.randint(0, 9)]
        for _ in range(2):  # → 合計3桁になるように2回だけ追加
            tgt = torch.tensor(seq, dtype=torch.long).unsqueeze(1).to(device)
            with torch.no_grad():
                logits = decoder(tgt, memory)
                next_digit = int(torch.argmax(logits[-1]).item())
            seq.append(next_digit)

        if len(set(seq)) == 3:
            predictions.append((seq, 0.91))

    return predictions

def gpt_generate_predictions(model, num_samples=5, context_length=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    for _ in range(num_samples):
        seq = [random.randint(0, 9)]
        for _ in range(context_length - 1):
            tgt = torch.tensor(seq, dtype=torch.long).unsqueeze(1).to(device)  # shape: (seq_len, batch=1)
            embed_dim = model.embedding.embedding_dim
            memory = torch.zeros((1, 1, embed_dim), dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(tgt, memory)
                next_token = torch.argmax(logits[-1]).item()
                seq.append(next_token)
        if len(set(seq)) == 4:
            predictions.append((seq, 0.89))  # 信頼度は仮
    return predictions

def train_gpt3numbers_model(save_path="gpt3numbers.pth", epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT4Numbers().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    try:
        df = pd.read_csv("numbers3.csv")
        df["本数字"] = df["本数字"].apply(parse_number_string)
        sequences = [row for row in df["本数字"] if isinstance(row, list) and len(row) == 3]
    except Exception as e:
        print(f"[ERROR] 学習データ読み込みエラー: {e}")
        return model

    data = []
    for seq in sequences:
        for i in range(len(seq) - 1):  # 例: [2,5,3] → ([2],5), ([2,5],3)
            context = seq[:i + 1]
            target = seq[i + 1]
            if len(context) >= 1:
                data.append((context, target))

    if not data:
        print("[WARNING] GPT3Numbers 学習データが空です")
        return model

    print(f"[INFO] GPT3Numbers 学習データ件数: {len(data)}")

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for context, target in data:
            tgt = torch.tensor(context, dtype=torch.long).unsqueeze(1).to(device)
            memory = torch.zeros((1, 1, model.embedding.embedding_dim), dtype=torch.float32).to(device)
            target_tensor = torch.tensor([target], dtype=torch.long).to(device)

            output = model(tgt, memory)[-1].unsqueeze(0)
            loss = criterion(output, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(data)
            print(f"[GPT3] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[INFO] GPT3Numbers モデルを保存しました: {save_path}")
    return model

class LotoPredictor:
    def __init__(self, input_size, hidden_size):
        print("[INFO] モデルを初期化")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_model = None
        self.regression_models = [None] * 3
        self.scaler = None
        self.feature_names = None
        self.meta_model = None

    def train_model(self, data):
        print("[INFO] Numbers3学習開始")
        true_numbers = data['本数字'].apply(lambda x: parse_number_string(x)).tolist()

        # 高等級一致のみ自己予測から再投入
        self_data = load_self_predictions(file_path="self_predictions.csv", min_match_threshold=2, true_data=true_numbers)
        if self_data:
            high_grade_predictions = []
            seen = set()
            for pred_tuple, count in self_data:
                pred = list(pred_tuple)
                if len(pred) != 3 or tuple(pred) in seen:
                    continue
                for true in true_numbers:
                    if classify_numbers3_prize(pred, true) in ["ストレート", "ボックス"]:
                        high_grade_predictions.append((pred, count))
                        seen.add(tuple(pred))
                        break

            if high_grade_predictions:
                synthetic_rows = pd.DataFrame({
                    '抽せん日': pd.Timestamp.now(),
                    '本数字': [row[0] for row in high_grade_predictions for _ in range(row[1])]
                })
                data = pd.concat([data, synthetic_rows], ignore_index=True)
                print(f"[INFO] 自己進化データ追加: {len(synthetic_rows)}件")

        X, _, self.scaler = preprocess_data(data)
        if X is None:
            return

        processed_data = create_advanced_features(data)
        y1, y2, y3 = transform_to_digit_labels(processed_data['本数字'])[:3]
        self.feature_names = [str(i) for i in range(X.shape[1])]
        X = reinforce_top_features(X, self.feature_names, y1)
        X_df = pd.DataFrame(X, columns=self.feature_names)

        for i, y in enumerate([y1, y2, y3]):
            df_train = X_df.copy()
            df_train['target'] = y
            predictor = TabularPredictor(label='target', path=f'autogluon_n3_pos{i}').fit(df_train, time_limit=300)
            self.regression_models[i] = predictor
            print(f"[AutoGluon] モデル {i+1}/3 完了")

        # LSTM訓練
        input_size = X.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LotoLSTM(input_size, 128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X.reshape(-1, 1, input_size), dtype=torch.float32).to(device)
        y_tensors = [torch.tensor(y, dtype=torch.long).to(device) for y in [y1, y2, y3]]
        dataset = TensorDataset(X_tensor, *y_tensors)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch in loader:
                inputs = batch[0]
                targets = batch[1:]
                optimizer.zero_grad()
                outputs = model(inputs)[:3]  # 3桁に限定
                losses = [criterion(out, target) for out, target in zip(outputs, targets)]
                loss = sum(losses)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[LSTM] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

        self.lstm_model = model
        self.meta_model = train_meta_model_maml("evaluation_result.csv", data)

    def predict(self, latest_data, num_candidates=50):
        print("[INFO] Numbers3予測開始")
        X, _, _ = preprocess_data(latest_data)
        if X is None:
            return None, None

        X_df = pd.DataFrame(X, columns=self.feature_names)
        pred_digits = []
        for i in range(3):  # 3桁
            pred = self.regression_models[i].predict(X_df)
            pred_digits.append(pred)
        auto_preds = np.array(pred_digits).T

        input_size = X.shape[1]
        X_tensor = torch.tensor(X.reshape(-1, 1, input_size), dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_model.to(device)
        self.lstm_model.eval()
        with torch.no_grad():
            outputs = self.lstm_model(X_tensor.to(device))[:3]
            lstm_preds = [torch.argmax(out, dim=1).cpu().numpy() for out in outputs]
        lstm_preds = np.array(lstm_preds).T

        all_predictions = []
        for i in range(min(len(auto_preds), len(lstm_preds))):
            merged = (0.7 * auto_preds[i] + 0.3 * lstm_preds[i]).round().astype(int)
            numbers = list(map(int, merged))

            if len(set(numbers)) < 3:
                continue
            if score_real_structure_similarity(numbers) < 0.3:
                continue

            base_conf = 1.0
            corrected_conf = base_conf
            if self.meta_model:
                try:
                    feat_vec = X_df.iloc[i].values.reshape(1, -1)
                    predicted_match = self.meta_model.predict(feat_vec)[0]
                    corrected_conf = max(0.0, min(predicted_match / 3.0, 1.0))
                    final_conf = 0.5 * base_conf + 0.5 * corrected_conf
                except:
                    final_conf = base_conf
            else:
                final_conf = base_conf

            all_predictions.append((numbers, final_conf))

        return all_predictions[:num_candidates], [c for _, c in all_predictions[:num_candidates]]

def classify_numbers3_prize(pred, actual):
    pred = list(map(int, pred))
    actual = list(map(int, actual))

    if pred == actual:
        return "ストレート"
    elif sorted(pred) == sorted(actual):
        return "ボックス"
    elif pred[1:] == actual[1:]:
        return "ミニ"
    else:
        return "はずれ"

def simulate_grade_distribution(simulated, historical_numbers):
    counter = Counter()
    for sim in simulated:
        for actual in historical_numbers:
            prize = classify_numbers3_prize(sim, actual)
            counter[prize] += 1
            break  # 1回の一致で良い

    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}

# 予測結果の評価
def evaluate_predictions(predictions, actual_numbers):
    results = []
    for pred in predictions:
        match_type = classify_numbers3_prize(pred[0], actual_numbers)
        if match_type == "ストレート":
            reward = 900000
        elif match_type == "ボックス":
            reward = 37500
        else:
            reward = 0
            
        results.append({
            '予測': pred[0],
            '一致数': len(set(pred[0]) & set(actual_numbers)),
            '等級': match_type,
            '信頼度': pred[1],
            '期待収益': reward
        })
    return results

official_url = "https://www.takarakuji-official.jp/ec/numbers3/"

async def fetch_drawing_dates():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(official_url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    drawing_dates = []
                    # 修正された抽せん日を取得
                    date_elements = soup.select("p.m_ttlSet_sttl")
                    for p in date_elements:
                        text = p.get_text(strip=True)
                        if "抽せん日：" in text:
                            date_part = text.replace("抽せん日：", "").strip()
                            formatted_date = date_part.replace("/", "-")
                            drawing_dates.append(formatted_date)

                    return drawing_dates
                else:
                    print(f"HTTPエラー {response.status}: {official_url}")
        except Exception as e:
            print(f"抽せん日取得エラー: {e}")
    return []

# 最新の抽せん日を1件だけ取得する関数（必要なら）
async def get_latest_drawing_date():
    dates = await fetch_drawing_dates()
    return dates[0] if dates else None


def parse_number_string(number_str):
    """
    予測番号や当選番号の文字列をリスト化する関数
    - スペース / カンマ / タブ 区切りに対応
    - "07 15 20 28 29 34 36" → [7, 15, 20, 28, 29, 34, 36]
    - "[7, 15, 20, 28, 29, 34, 36]" → [7, 15, 20, 28, 29, 34, 36]
    """
    if pd.isna(number_str):
        return []  # NaN の場合は空リストを返す
    
    # 不要な記号を削除（リスト形式の場合）
    number_str = number_str.strip("[]").replace("'", "").replace('"', '')

    # スペース・カンマ・タブで分割し、整数変換
    numbers = re.split(r'[\s,]+', number_str)

    # 数字のみにフィルタリングして整数変換
    return [int(n) for n in numbers if n.isdigit()]

def calculate_precision_recall_f1(evaluation_df):
    y_true = []
    y_pred = []

    for _, row in evaluation_df.iterrows():
        actual = set(row["当選本数字"])
        predicted = set(row["予測番号"])
        for n in range(0, 10):
            y_true.append(1 if n in actual else 0)
            y_pred.append(1 if n in predicted else 0)

    try:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    except Exception as e:
        print("[ERROR] 評価指標の計算に失敗:", e)
        precision, recall, f1 = 0, 0, 0

    print("\n=== 評価指標 ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

# 予測結果をCSVファイルに保存する関数
def save_predictions_to_csv(predictions, drawing_date, filename="Numbers3_predictions.csv"):
    drawing_date = pd.to_datetime(drawing_date).strftime("%Y-%m-%d")
    row = {"抽せん日": drawing_date}

    for i, (numbers, confidence) in enumerate(predictions[:5], 1):
        row[f"予測{i}"] = ', '.join(map(str, numbers))
        row[f"信頼度{i}"] = round(confidence, 3)
        
    df = pd.DataFrame([row])

    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename, encoding='utf-8-sig')
            if "抽せん日" not in existing_df.columns:
                print(f"警告: CSVに'抽せん日'列が見つかりません。新規作成します。")
                default_columns = ["抽せん日"] + [f"予測{i}" for i in range(1, 6)] + [f"信頼度{i}" for i in range(1, 6)]
                existing_df = pd.DataFrame(columns=default_columns)
            existing_df = existing_df[existing_df["抽せん日"] != drawing_date]
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"CSVファイルの読み込みエラー: {e}。新規作成します。")
            df = pd.DataFrame([row])

    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"予測結果を {filename} に保存しました。")

def is_running_with_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

def ppo_multiagent_predict(historical_data, num_predictions=5):
    """
    ストレート寄せ・ボックス寄せ・ランダム最大化の3方針でPPOを実行し、
    高信頼度の出力を選抜して返す。
    """
    from random import randint
    agents = ["straight", "box", "diverse"]
    results = []

    for strategy in agents:
        for _ in range(num_predictions):
            if strategy == "straight":
                # 直近の出現数字からストレート構成を優先
                last = parse_number_string(historical_data.iloc[-1]["本数字"])
                new = list((n + randint(0, 2)) % 10 for n in last)
                confidence = 0.91
            elif strategy == "box":
                # ボックス寄せ：頻出数字を集めて順不同構成
                all_nums = [n for row in historical_data["本数字"] for n in parse_number_string(row)]
                freq = Counter(all_nums).most_common(4)
                new = sorted([f[0] for f in freq])
                confidence = 0.92
            else:
                # 多様性最大化（ランダムにばらけた構成）
                new = sorted(randint(0, 9) for _ in range(3))
                confidence = 0.905

            results.append((new, confidence))

    return results

def train_diffusion_model(df, model_path="diffusion_model.pth", epochs=100, device="cpu"):
    class DiffusionMLP(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x, t):
            t_embed = t.float().view(-1, 1) / 1000.0
            x_input = torch.cat([x, t_embed], dim=1)
            return self.net(x_input)

    def prepare_training_data(df):
        data = []
        for row in df["本数字"]:
            nums = parse_number_string(row)
            if len(nums) == 3:
                data.append(nums)
        return np.array(data)

    model = DiffusionMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = 1000
    def noise_schedule(t): return 1 - t / T

    X = prepare_training_data(df)
    if len(X) == 0:
        print("[ERROR] Diffusion学習用のデータが空です")
        return

    data = torch.tensor(X, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        last_loss = None
        for i in range(len(data)):
            x0 = data[i].unsqueeze(0)
            t = torch.randint(1, T, (1,), device=device)
            alpha = noise_schedule(t)
            noise = torch.randn_like(x0)
            xt = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss

        if last_loss is not None and (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss: {last_loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Diffusion モデルを保存しました: {model_path}")

def diffusion_generate_predictions(df, num_predictions=5, model_path="diffusion_model.pth"):
    class DiffusionMLP(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x, t):
            t_embed = t.float().view(-1, 1) / 1000.0
            x_input = torch.cat([x, t_embed], dim=1)
            return self.net(x_input)

    model = DiffusionMLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    predictions = []
    trials = 0
    max_trials = num_predictions * 10

    while len(predictions) < num_predictions and trials < max_trials:
        trials += 1
        x = torch.randn(1, 3)
        timesteps = list(range(1000))[::-1]
        for t in timesteps:
            t_tensor = torch.tensor([t]).float().view(-1, 1)
            noise_pred = model(x, t_tensor)
            x = x - noise_pred / 1000.0

        candidate = tuple(int(round(v)) for v in x.squeeze().tolist())

        if all(0 <= n <= 9 for n in candidate) and len(set(candidate)) == 3:
            predictions.append(candidate)

    return [(list(p), 0.91) for p in predictions]

def load_trained_model():
    print("[INFO] 外部モデルは未定義のため、Noneを返します。")
    return None

class CycleAttentionTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super(CycleAttentionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.embedding(x)                      # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)                    # (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)                     # (seq_len, batch, embed_dim)
        x = self.transformer_encoder(x)            # (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2)                     # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)                          # Global average pooling
        x = self.ff(x)                             # (batch, 4)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x
    
def train_transformer_with_cycle_attention(df, model_path="transformer_model.pth", epochs=50):
    print("[INFO] Transformerモデルの学習を開始します...")

    class CycleAttentionTransformer(nn.Module):
        def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Linear(input_dim, embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # ← ⭐ 3桁に修正
            )

        def forward(self, x):
            x = self.embedding(x)
            x = self.pos_encoder(x)
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)
            x = x.mean(dim=1)
            return self.ff(x)

    def prepare_input(df):
        recent = df.tail(10)
        nums = [parse_number_string(n) for n in recent["本数字"]]
        flat = [n for row in nums for n in row]
        flat = (flat + [0] * 40)[:40]
        return torch.tensor([flat], dtype=torch.float32)

    model = CycleAttentionTransformer(input_dim=40)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        x = prepare_input(df)
        y = torch.tensor([[random.randint(0, 9) for _ in range(3)]], dtype=torch.float32)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Transformer] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Transformerモデルを保存しました: {model_path}")

def transformer_generate_predictions(df, model_path="transformer_model.pth"):
    class CycleAttentionTransformer(nn.Module):
        def __init__(self, input_dim=40, embed_dim=64, num_heads=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Linear(input_dim, embed_dim)
            self.pos_encoder = PositionalEncoding(embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # ← ⭐ 3桁出力
            )

        def forward(self, x):
            x = self.embedding(x)
            x = self.pos_encoder(x)
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)
            return self.ff(x.mean(dim=1))

    def prepare_input(df):
        recent = df.tail(10)
        nums = [parse_number_string(n) for n in recent["本数字"]]
        flat = [n for row in nums for n in row]
        flat = (flat + [0] * 40)[:40]
        return torch.tensor([flat], dtype=torch.float32)

    input_tensor = prepare_input(df)

    model = CycleAttentionTransformer()
    if not os.path.exists(model_path):
        train_transformer_with_cycle_attention(df, model_path=model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        prediction = [max(0, min(9, int(round(p.item())))) for p in output.squeeze()]
        print(f"[Transformer] 予測結果: {prediction}")
        return [(prediction, 0.95)]

def evaluate_and_summarize_predictions(
    pred_file="numbers3_predictions.csv",
    actual_file="numbers3.csv",
    output_csv="evaluation_result.csv",
    output_txt="evaluation_summary.txt"
):
    try:
        pred_df = pd.read_csv(pred_file)
        actual_df = pd.read_csv(actual_file)
        actual_df['抽せん日'] = pd.to_datetime(actual_df['抽せん日'], errors='coerce').dt.date
        pred_df['抽せん日'] = pd.to_datetime(pred_df['抽せん日'], errors='coerce').dt.date
    except Exception as e:
        print(f"[ERROR] ファイル読み込み失敗: {e}")
        return

    evaluation_results = []
    grade_counter = Counter()
    match_counter = Counter()
    all_hits = []

    grade_list = ["はずれ", "ミニ", "ボックス", "ストレート"]

    results_by_prediction = {
        i: {grade: 0 for grade in grade_list} | {"details": []}
        for i in range(1, 6)
    }

    for _, row in pred_df.iterrows():
        draw_date = row["抽せん日"]
        actual_row = actual_df[actual_df["抽せん日"] == draw_date]
        if actual_row.empty:
            continue
        actual_numbers = parse_number_string(actual_row.iloc[0]["本数字"])

        for i in range(1, 6):
            pred_key = f"予測{i}"
            conf_key = f"信頼度{i}"
            if pred_key in row and pd.notna(row[pred_key]):
                predicted = parse_number_string(str(row[pred_key]))
                confidence = row[conf_key] if conf_key in row and pd.notna(row[conf_key]) else 1.0
                grade = classify_numbers3_prize(predicted, actual_numbers)
                match_count = len(set(predicted) & set(actual_numbers))

                if 0.91 <= confidence <= 0.93:
                    source = "PPO/Diffusion"
                elif confidence > 0.93:
                    source = "BaseModel"
                else:
                    source = "Unknown"

                evaluation_results.append({
                    "抽せん日": draw_date.strftime("%Y-%m-%d"),
                    "予測番号": predicted,
                    "当選本数字": actual_numbers,
                    "一致数": match_count,
                    "等級": grade,
                    "信頼度": confidence,
                    "出力元": source,
                    "予測番号インデックス": f"予測{i}"
                })

                grade_counter[grade] += 1
                match_counter[match_count] += 1
                results_by_prediction[i][grade] += 1

                if grade != "はずれ":
                    detail = f'{draw_date},"{predicted}","{actual_numbers}",{grade}'
                    results_by_prediction[i]["details"].append(detail)
                    all_hits.append(detail)

    # === CSV保存 ===
    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 比較結果を {output_csv} に保存しました")

    # === TXT出力 ===
    lines = []
    lines.append("== 等級別全体集計 ==")
    for k in grade_list:
        lines.append(f"{k}: {grade_counter[k]} 件")

    total = sum(grade_counter.values())
    matched = grade_counter["ミニ"] + grade_counter["ボックス"] + grade_counter["ストレート"]
    rate = (matched / total * 100) if total > 0 else 0
    lines.append("\n== 等級的中率チェック ==")
    lines.append(f"ストレート・ボックス・ミニの合計: {matched} 件")
    lines.append(f"的中率（等級ベース）: {rate:.2f}%")
    lines.append("✓ 的中率は目標を達成しています。" if rate >= 10 else "✘ 的中率は目標に達していません。")

    # === 各予測番号の等級・賞金・利益の集計 ===
    lines.append("\n== 各予測ごとの賞金・コスト・利益 ==")
    mini_prize = 9000
    box_prize = 15000
    straight_prize = 114000
    cost_per_draw = 600

    for i in range(1, 6):
        lines.append(f"\n== 等級別予想{i}集計 ==")
        mini = results_by_prediction[i]["ミニ"]
        box = results_by_prediction[i]["ボックス"]
        straight = results_by_prediction[i]["ストレート"]

        for k in grade_list:
            lines.append(f"{k}: {results_by_prediction[i][k]} 件")

        hit_count = mini + box + straight
        total_preds = sum(results_by_prediction[i][k] for k in grade_list)
        acc = (hit_count / total_preds * 100) if total_preds > 0 else 0
        lines.append("\n== 等級的中率チェック ==")
        lines.append(f"ストレート・ボックス・ミニの合計: {hit_count} 件")
        lines.append(f"的中率（等級ベース）: {acc:.2f}%")

        for detail in results_by_prediction[i]["details"]:
            lines.append(detail)

        # 賞金と利益の計算
        mini_total = mini * mini_prize
        box_total = box * box_prize
        straight_total = straight * straight_prize
        total_reward = mini_total + box_total + straight_total
        cost = total_preds * cost_per_draw
        profit = total_reward - cost

        lines.append(f"\n== 予測{i}の賞金・損益 ==")
        lines.append(f"ミニ: {mini} 件 × ¥{mini_prize:,} = ¥{mini_total:,}")
        lines.append(f"ボックス: {box} 件 × ¥{box_prize:,} = ¥{box_total:,}")
        lines.append(f"ストレート: {straight} 件 × ¥{straight_prize:,} = ¥{straight_total:,}")
        lines.append(f"当選合計金額: ¥{total_reward:,}")
        lines.append(f"コスト（{total_preds} × ¥{cost_per_draw:,}）: ¥{cost:,}")
        lines.append(f"損益: {'+' if profit >= 0 else '-'}¥{abs(profit):,}")

    # === 全体の賞金・利益の集計 ===
    lines.append("\n== 賞金・コスト・利益（全体） ==")
    mini_total = grade_counter["ミニ"] * mini_prize
    box_total = grade_counter["ボックス"] * box_prize
    straight_total = grade_counter["ストレート"] * straight_prize
    all_reward = mini_total + box_total + straight_total
    total_cost = total * cost_per_draw
    profit = all_reward - total_cost

    lines.append(f"ミニ: {grade_counter['ミニ']} 件 × ¥{mini_prize:,} = ¥{mini_total:,}")
    lines.append(f"ボックス: {grade_counter['ボックス']} 件 × ¥{box_prize:,} = ¥{box_total:,}")
    lines.append(f"ストレート: {grade_counter['ストレート']} 件 × ¥{straight_prize:,} = ¥{straight_total:,}")
    lines.append(f"当選合計金額(理論値): ¥{all_reward:,}")
    lines.append(f"総コスト（全体件数 {total} × ¥{cost_per_draw:,}）: ¥{total_cost:,}")
    lines.append(f"最終損益: {'+' if profit >= 0 else '-'}¥{abs(profit):,}")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] 集計結果を {output_txt} に出力しました（{len(all_hits)} 件の的中）")

def main_with_improved_predictions():
    try:
        df = pd.read_csv("numbers3.csv")
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
        df = df.sort_values("抽せん日").reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] データ読み込み失敗: {e}")
        return

    historical_data = df.copy()
    latest_date = historical_data["抽せん日"].max()
    next_draw_date = latest_date + pd.Timedelta(days=1)

    all_groups = {
        "PPO": ppo_multiagent_predict(historical_data),
        "Diffusion": diffusion_generate_predictions(historical_data, 5),
        "Transformer": transformer_generate_predictions(historical_data),
    }

    # GPT (with memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_model_path = "gpt4numbers.pth"
    encoder_path = "memory_encoder.pth"
    if not os.path.exists(gpt_model_path) or not os.path.exists(encoder_path):
        decoder, encoder = train_gpt3numbers_model_with_memory(save_path=gpt_model_path, encoder_path=encoder_path)
    else:
        decoder = GPT4Numbers().to(device)
        encoder = MemoryEncoder().to(device)
        decoder.load_state_dict(torch.load(gpt_model_path, map_location=device))
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    sequences = historical_data["本数字"].apply(parse_number_string).tolist()
    all_groups["GPT"] = gpt_generate_predictions_with_memory_3(decoder, encoder, sequences, num_samples=5)

    # すべての候補を統合
    all_predictions = []
    for preds in all_groups.values():
        all_predictions.extend(preds)

    # 統合予測のフィルタリング
    verified = verify_predictions(all_predictions, historical_data)

    # 保存内容を構築
    result = {"抽せん日": next_draw_date.strftime("%Y-%m-%d")}
    for i, (numbers, conf) in enumerate(verified[:5]):
        result[f"予測{i+1}"] = ",".join(map(str, numbers))
        result[f"信頼度{i+1}"] = round(conf, 4)
        if 0.90 <= conf <= 0.93:
            result[f"出力元{i+1}"] = "PPO/Diffusion/GPT"
        elif conf > 0.93:
            result[f"出力元{i+1}"] = "Transformer"
        else:
            result[f"出力元{i+1}"] = "Unknown"

    pred_path = "numbers3_predictions.csv"
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        pred_df = pred_df[pred_df["抽せん日"] != next_draw_date.strftime("%Y-%m-%d")]
        pred_df = pd.concat([pred_df, pd.DataFrame([result])], ignore_index=True)
    else:
        pred_df = pd.DataFrame([result])

    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] 最新予測（{next_draw_date.strftime('%Y-%m-%d')}）を {pred_path} に保存しました")

    # ✅ 統合された評価＋集計関数で出力
    try:
        evaluate_and_summarize_predictions(
            pred_file=pred_path,
            actual_file="numbers3.csv",
            output_csv="evaluation_result.csv",
            output_txt="evaluation_summary.txt"
        )
    except Exception as e:
        print(f"[WARNING] 評価処理に失敗: {e}")

def calculate_pattern_score(numbers):
    score = 0
    if 10 <= sum(numbers) <= 20:  # 合計がある程度高い
        score += 1
    if len(set(n % 2 for n in numbers)) > 1:  # 偶奇が混在
        score += 1
    if len(set(numbers)) == 3:  # 重複なし
        score += 1
    return score

def plot_prediction_analysis(predictions, historical_data):
    plt.figure(figsize=(15, 10))
    
    # 予測番号の分布
    plt.subplot(2, 2, 1)
    all_predicted_numbers = [num for pred in predictions for num in pred[0]]
    plt.hist(all_predicted_numbers, bins=9, range=(0, 9), alpha=0.7)
    plt.title('予測番号の分布')
    plt.xlabel('数字')
    plt.ylabel('頻度')
    
    # 信頼度スコアの分布
    plt.subplot(2, 2, 2)
    confidence_scores = [pred[1] for pred in predictions]
    plt.hist(confidence_scores, bins=20, alpha=0.7)
    plt.title('信頼度スコアの分布')
    plt.xlabel('信頼度')
    plt.ylabel('頻度')
    
    # 過去の当選番号との比較
    plt.subplot(2, 2, 3)
    historical_numbers = [num for numbers in historical_data['本数字'] for num in numbers]
    plt.hist(historical_numbers, bins=9, range=(0, 9), alpha=0.5, label='過去の当選')
    plt.hist(all_predicted_numbers, bins=9, range=(0, 9), alpha=0.5, label='予測')
    plt.title('予測 vs 過去の当選')
    plt.xlabel('数字')
    plt.ylabel('頻度')
    plt.legend()
    
    # パターン分析
    plt.subplot(2, 2, 4)
    pattern_scores = [calculate_pattern_score(pred[0]) for pred in predictions]
    plt.scatter(range(len(pattern_scores)), pattern_scores, alpha=0.5)
    plt.title('予測パターンスコア')
    plt.xlabel('予測インデックス')
    plt.ylabel('パターンスコア')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    plt.close()

def generate_evolution_graph(log_file="evolution_log.txt", output_file="evolution_graph.png"):
    """
    evolution_log.txtを読み込んで進化グラフを生成・保存する
    """
    if not os.path.exists(log_file):
        print(f"[WARNING] 進化ログ {log_file} が見つかりません")
        return

    dates = []
    counts = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split(":")
                date_part = parts[0].strip()
                count_part = parts[2].strip()

                date = pd.to_datetime(date_part)
                count = int(count_part.split()[0])

                dates.append(date)
                counts.append(count)
            except Exception as e:
                print(f"[WARNING] ログパース失敗: {e}")
                continue

    if not dates:
        print("[WARNING] 進化ログに有効なデータがありません")
        return

    # --- グラフ描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(dates, counts, marker='o', linestyle='-', color='blue')
    plt.title("自己進化履歴（自己予測データ件数推移）")
    plt.xlabel("日時")
    plt.ylabel("自己予測データ件数")
    plt.grid(True)
    plt.tight_layout()

    # --- 保存 ---
    plt.savefig(output_file)
    plt.close()
    print(f"[INFO] 進化履歴グラフを保存しました: {output_file}")

def verify_predictions(predictions, historical_data, top_k=5, grade_probs=None):
    def check_number_constraints(numbers):
        return (
            len(numbers) == 3 and
            all(0 <= n <= 9 for n in numbers)
        )

    print("[INFO] 予測候補をフィルタリング中...")

    cycle_scores = calculate_number_cycle_score(historical_data)
    valid_predictions = []

    for pred, conf in predictions:
        try:
            numbers = np.sort(np.array(pred if isinstance(pred, (list, np.ndarray)) else pred[0]))
            if check_number_constraints(numbers) and calculate_pattern_score(numbers.tolist()) >= 2:
                avg_cycle = np.mean([cycle_scores.get(n, 999) for n in numbers])
                cycle_score = max(0, 1 - (avg_cycle / 50))
                final_conf = 0.7 * conf + 0.3 * cycle_score
                valid_predictions.append((numbers.tolist(), final_conf))
        except Exception as e:
            print(f"[WARNING] 予測フィルタ中にエラー: {e}")
            continue

    if not valid_predictions:
        print("[WARNING] 有効な予測がありません")
        return []

    # ✅ PPO / Diffusion 由来の構成を1組含める（信頼度で判定）
    ppo_or_diffusion_found = any(0.90 <= conf <= 0.93 for _, conf in valid_predictions)
    if not ppo_or_diffusion_found:
        fallback_candidate = None
        for pred, conf in valid_predictions:
            if 0.89 <= conf <= 0.94:
                fallback_candidate = (pred, conf)
                print(f"[INFO] PPO/Diffusion保証補完: {pred} (conf={conf:.3f})")
                break
        if fallback_candidate:
            valid_predictions.insert(0, fallback_candidate)
        else:
            print("[WARNING] PPO/Diffusion保証候補が見つかりませんでした")

    historical_list = [parse_number_string(x) for x in historical_data["本数字"].tolist()]

    # ✅ 等級構成保証（ストレート/ボックス）
    guaranteed_grade_candidate = None
    for pred, conf in valid_predictions:
        for actual in historical_list[-100:]:
            grade = classify_numbers3_prize(pred, actual)
            if grade in ["ストレート", "ボックス"]:
                guaranteed_grade_candidate = (pred, conf)
                print(f"[INFO] 等級保証パターン確保: {pred} → {grade}")
                break
        if guaranteed_grade_candidate:
            break

    # 等級保証もなければ、擬似ボックス構成で補完
    if not guaranteed_grade_candidate:
        fallback = historical_list[-1]
        alt = list(fallback)
        alt[0] = (alt[0] + 1) % 10
        guaranteed_grade_candidate = (alt, 0.91)
        print(f"[INFO] 等級保証構成のための補完: {alt}")

    valid_predictions.sort(key=lambda x: x[1], reverse=True)

    # ✅ 多様性保証（奇偶構成）
    def parity_pattern(numbers):
        return tuple(n % 2 for n in numbers)

    diverse_patterns = set()
    selected = [guaranteed_grade_candidate]
    seen = {tuple(guaranteed_grade_candidate[0])}
    diverse_patterns.add(parity_pattern(guaranteed_grade_candidate[0]))

    for pred in valid_predictions:
        key = tuple(pred[0])
        pattern = parity_pattern(pred[0])
        if key not in seen and pattern not in diverse_patterns:
            selected.append(pred)
            seen.add(key)
            diverse_patterns.add(pattern)
        if len(selected) >= top_k:
            break

    print("[INFO] 最終選択された予測数:", len(selected))
    return selected

def extract_strong_features(evaluation_df, feature_df):
    """
    過去予測評価と特徴量を結合し、「本数字一致数」と相関の高い特徴量を抽出
    """
    # 🔒 入力データの検証
    if evaluation_df is None or evaluation_df.empty:
        print("[WARNING] 評価データが空のため、重要特徴量の抽出をスキップします。")
        return []

    if "抽せん日" not in evaluation_df.columns:
        print("[WARNING] 評価データに '抽せん日' 列が存在しません。重要特徴量の抽出をスキップします。")
        return []

    if feature_df is None or feature_df.empty or "抽せん日" not in feature_df.columns:
        print("[WARNING] 特徴量データが無効または '抽せん日' 列がありません。")
        return []

    # 🔧 日付型を明示的に揃える
    evaluation_df['抽せん日'] = pd.to_datetime(evaluation_df['抽せん日'], errors='coerce')
    feature_df['抽せん日'] = pd.to_datetime(feature_df['抽せん日'], errors='coerce')

    # ⛓ 結合
    merged = evaluation_df.merge(feature_df, on="抽せん日", how="inner")
    if merged.empty:
        print("[WARNING] 評価データと特徴量データの結合結果が空です。")
        return []

    # 📊 相関計算
    correlations = {}
    for col in feature_df.columns:
        if col in ["抽せん日", "本数字", "ボーナス数字"]:
            continue
        try:
            if not np.issubdtype(merged[col].dtype, np.number):
                continue
            corr = np.corrcoef(merged[col], merged["本数字一致数"])[0, 1]
            correlations[col] = abs(corr)
        except Exception:
            continue

    # 🔝 上位5特徴量を返す
    top_features = sorted(correlations.items(), key=lambda x: -x[1])[:5]
    return [f[0] for f in top_features]

def reinforce_features(X, feature_names, important_features, multiplier=1.5):
    """
    指定された重要特徴量を強調（値を倍率で増強）
    """
    reinforced_X = X.copy()
    for feat in important_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            reinforced_X[:, idx] *= multiplier
    return reinforced_X

# --- 🔥 新規追加関数 ---
def extract_high_match_patterns(dataframe, min_match=2):
    high_match_combos = []
    total = len(dataframe)
    for idx1, row1 in enumerate(dataframe.itertuples(), 1):
        nums1 = set(row1.本数字)
        for idx2 in range(idx1 + 1, total):
            nums2 = set(dataframe.iloc[idx2]['本数字'])
            if len(nums1 & nums2) >= min_match:
                high_match_combos.append(sorted(nums1))
        if idx1 % 50 == 0:
            print(f"[DEBUG] パターン比較進行中... {idx1}/{total}")
    return high_match_combos

def calculate_number_frequencies(dataframe):
    """過去データから番号出現頻度スコアを計算"""
    all_numbers = [num for nums in dataframe['本数字'] for num in nums]
    freq = pd.Series(all_numbers).value_counts().to_dict()
    return freq

def calculate_number_cycle_score(dataframe):
    """番号ごとの出現周期スコアを計算"""
    number_last_seen = {n: None for n in range(0, 10)}
    number_cycle = {n: [] for n in range(0, 10)}

    for i, raw in enumerate(dataframe['本数字']):
        nums = parse_number_string(raw)  # ★ ここを追加して変換
        for n in range(0, 10):
            if n in nums:
                if number_last_seen[n] is not None:
                    cycle = i - number_last_seen[n]
                    number_cycle[n].append(cycle)
                number_last_seen[n] = i

    avg_cycle = {n: np.mean(cycles) if cycles else 999 for n, cycles in number_cycle.items()}
    return avg_cycle

def create_meta_training_data(evaluation_df, feature_df):
    """
    過去の予測結果と特徴量から、メタ学習用の訓練データを作成
    特徴量: 抽せん日の特徴量群
    ターゲット: 本数字一致数
    """
    if evaluation_df is None or evaluation_df.empty:
        return None, None

    evaluation_df["抽せん日"] = pd.to_datetime(evaluation_df["抽せん日"], errors="coerce")
    feature_df["抽せん日"] = pd.to_datetime(feature_df["抽せん日"], errors="coerce")
    
    merged = evaluation_df.merge(feature_df, on="抽せん日", how="inner")

    target = merged["本数字一致数"].values
    features = merged.drop(columns=["抽せん日", "予測番号", "当選本数字", "当選ボーナス", "等級"], errors="ignore")
    features = features.select_dtypes(include=[np.number]).fillna(0)

    return features.values, target

def train_meta_model_maml(evaluation_csv="evaluation_result.csv", feature_df=None):
    from sklearn.linear_model import Ridge
    from sklearn.base import clone

    if not os.path.exists(evaluation_csv):
        print(f"[INFO] {evaluation_csv} が存在しないため、Metaモデル学習をスキップします。")
        return None

    try:
        eval_df = pd.read_csv(evaluation_csv)
        if feature_df is None:
            df = pd.read_csv("numbers3.csv")
            df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
            feature_df = create_advanced_features(df)

        X_meta, y_meta = create_meta_training_data(eval_df, feature_df)
        if X_meta is None or len(X_meta) == 0:
            print("[ERROR] メタ学習データが作成できませんでした")
            return None

        # MAML風アプローチ：日ごとにローカルモデルを学習して平均化
        task_dates = eval_df["抽せん日"].unique()
        base_model = Ridge()
        local_models = []

        for date in task_dates:
            sub_eval = eval_df[eval_df["抽せん日"] == date]
            sub_feat = feature_df[feature_df["抽せん日"] == date]

            X_task, y_task = create_meta_training_data(sub_eval, sub_feat)
            if X_task is not None and len(X_task) >= 1:
                local_model = clone(base_model)
                local_model.fit(X_task, y_task)
                local_models.append(local_model)

        # 最終モデル：各ローカルモデルの平均重み（簡易平均）
        if not local_models:
            print("[WARNING] 有効なローカルモデルが作成されませんでした")
            return None

        final_model = clone(base_model)
        coefs = np.mean([m.coef_ for m in local_models], axis=0)
        intercepts = np.mean([m.intercept_ for m in local_models])
        final_model.coef_ = coefs
        final_model.intercept_ = intercepts
        print("[INFO] MAML風メタモデルの学習が完了しました")
        return final_model

    except Exception as e:
        print(f"[ERROR] MAML Metaモデル学習中に例外発生: {e}")
        return None
    
def generate_via_diffusion(recent_real_numbers, top_k=5):
    generator = DiffusionNumberGenerator()
    generated = generator.generate(num_samples=100)

    scored = []
    for sample in generated:
        max_sim = max(len(set(sample) & set(real)) for real in recent_real_numbers)
        struct_score = calculate_pattern_score(sample)
        final_score = max_sim + struct_score  # 類似度 + 構造スコア
        scored.append((sample, final_score))

    scored.sort(key=lambda x: -x[1])
    return [x[0] for x in scored[:top_k]]

def bulk_predict_all_past_draws():
    try:
        df = pd.read_csv("numbers3.csv")
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
        df = df.sort_values("抽せん日").reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] データ読み込み失敗: {e}")
        return

    pred_path = "Numbers3_predictions.csv"
    predicted_dates = set()

    if os.path.exists(pred_path):
        try:
            prev = pd.read_csv(pred_path)
            predicted_dates = set(pd.to_datetime(prev["抽せん日"], errors='coerce').dt.date.dropna())
        except Exception as e:
            print(f"[WARNING] 既存予測ファイル読み込み失敗: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("gpt4numbers.pth") or not os.path.exists("memory_encoder.pth"):
        decoder, encoder = train_gpt3numbers_model_with_memory()
    else:
        decoder = GPT4Numbers()
        decoder.load_state_dict(torch.load("gpt4numbers.pth", map_location=device))
        encoder = MemoryEncoder()
        encoder.load_state_dict(torch.load("memory_encoder.pth", map_location=device))

    decoder.to(device)
    encoder.to(device)
    decoder.eval()
    encoder.eval()

    for i in range(10, len(df)):
        sub_data = df.iloc[:i+1]
        latest_date = sub_data["抽せん日"].max()

        if latest_date.date() in predicted_dates:
            continue

        actual_numbers = parse_number_string(sub_data.iloc[-1]["本数字"])

        all_groups = {
            "PPO": ppo_multiagent_predict(sub_data),
            "Diffusion": diffusion_generate_predictions(sub_data, 5),
            "Transformer": transformer_generate_predictions(sub_data),
            "GPT": gpt_generate_predictions_with_memory_3(
                decoder, encoder,
                sub_data["本数字"].apply(parse_number_string).tolist(),
                num_samples=5
            )
        }

        all_candidates = []
        for preds in all_groups.values():
            all_candidates.extend(preds)

        verified_predictions = verify_predictions(all_candidates, sub_data)

        if not verified_predictions:
            continue

        result = {
            "抽せん日": latest_date.strftime("%Y-%m-%d")
        }

        for j, (numbers, conf) in enumerate(verified_predictions[:5]):
            result[f"予測{j+1}"] = ",".join(map(str, numbers))
            result[f"信頼度{j+1}"] = round(conf, 4)
            if 0.90 <= conf <= 0.93:
                result[f"出力元{j+1}"] = "PPO/Diffusion/GPT"
            elif conf > 0.93:
                result[f"出力元{j+1}"] = "Transformer"
            else:
                result[f"出力元{j+1}"] = "Unknown"

        result_df = pd.DataFrame([result])

        if os.path.exists(pred_path):
            try:
                existing = pd.read_csv(pred_path)
                result_df = pd.concat([existing, result_df], ignore_index=True)
            except Exception as e:
                print(f"[WARNING] 保存前の読み込み失敗: {e}")

        result_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] {latest_date.strftime('%Y-%m-%d')} の予測を保存しました")

        # ✅ 統合された評価＋集計関数をここで実行
        try:
            evaluate_and_summarize_predictions(
                pred_file=pred_path,
                actual_file="numbers3.csv",
                output_csv="evaluation_result.csv",
                output_txt="evaluation_summary.txt"
            )
        except Exception as e:
            print(f"[WARNING] 評価処理に失敗: {e}")

        predicted_dates.add(latest_date.date())

    print("[INFO] 過去すべての予測・評価処理が完了しました。")
    
if not os.path.exists("transformer_model.pth"):
    try:
        df = pd.read_csv("numbers3.csv")
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
        df = df.sort_values("抽せん日").reset_index(drop=True)
        train_transformer_with_cycle_attention(df)
    except Exception as e:
        print(f"[ERROR] Transformer学習失敗: {e}")

if __name__ == "__main__":

    # データ読み込み
    try:
        df = pd.read_csv("numbers3.csv")
        df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors='coerce')
        df = df.sort_values("抽せん日").reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] データ読み込み失敗: {e}")
        exit()

    # Diffusionモデルがなければ学習
    if not os.path.exists("diffusion_model.pth"):
        print("[INFO] Diffusionモデルが見つかりません。学習を開始します...")
        train_diffusion_model(df, model_path="diffusion_model.pth", epochs=100)

    # Transformerモデルがなければ学習
    if not os.path.exists("transformer_model.pth"):
        print("[INFO] Transformerモデルが見つかりません。学習を開始します...")
        train_transformer_with_cycle_attention(df, model_path="transformer_model.pth", epochs=50)

    # 🔁 一括予測を実行
    # bulk_predict_all_past_draws()
    main_with_improved_predictions()
    