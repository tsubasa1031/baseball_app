import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pybaseball import statcast
import warnings

# 警告を無視
warnings.filterwarnings('ignore')

def fetch_real_data():
    print("⚾ MLB公式データ(Statcast)を取得中... ")
    # データ量を削減するため、期間を1週間分にします
    df = statcast(start_dt='2024-05-01', end_dt='2024-05-07')
    print(f"✅ 取得完了: {len(df)} 球のデータを確保しました。")
    return df

def preprocess_data(df):
    print("⚙️ データを学習用に加工中...")
    
    # 1. 勝敗結果の作成
    game_results = df.groupby('game_pk').agg({
        'home_score': 'max',
        'away_score': 'max'
    }).reset_index()
    
    game_results['home_win_flag'] = (game_results['home_score'] > game_results['away_score']).astype(int)
    df = df.merge(game_results[['game_pk', 'home_win_flag']], on='game_pk', how='left')
    
    # 2. 基本特徴量の作成
    df['score_diff'] = df['home_score'] - df['away_score']
    df['is_top'] = (df['inning_topbot'] == 'Top').astype(int)
    df['on_1b'] = df['on_1b'].notnull().astype(int)
    df['on_2b'] = df['on_2b'].notnull().astype(int)
    df['on_3b'] = df['on_3b'].notnull().astype(int)

    # 3. 選手成績（OPS）の計算
    # 処理を軽量化するため、今回は簡易的に全期間の平均値を使用
    print("   打者・投手の成績を集計中...")

    def calculate_ops_simple(group):
        events = group['events']
        hits = events.isin(['single', 'double', 'triple', 'home_run']).sum()
        ab = (~events.isin(['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'intent_walk'])).sum()
        walks = events.isin(['walk', 'hit_by_pitch', 'intent_walk']).sum()
        tb = (events == 'single').sum() * 1 + (events == 'double').sum() * 2 + \
             (events == 'triple').sum() * 3 + (events == 'home_run').sum() * 4
        
        obp = (hits + walks) / (ab + walks) if (ab + walks) > 0 else 0.3
        slg = tb / ab if ab > 0 else 0.4
        return obp + slg

    # グループ化して計算（データ量が少ないのでこのまま処理）
    batter_ops = df.groupby('batter').apply(calculate_ops_simple).to_dict()
    pitcher_ops = df.groupby('pitcher').apply(calculate_ops_simple).to_dict()
    
    # マッピング（計算できなかった選手は平均値 0.720 で埋める）
    df['batter_ops'] = df['batter'].map(batter_ops).fillna(0.720)
    df['pitcher_opp_ops'] = df['pitcher'].map(pitcher_ops).fillna(0.720)

    # 特徴量選択
    feature_cols = [
        'score_diff', 'inning', 'is_top', 'outs_when_up', 
        'on_1b', 'on_2b', 'on_3b',
        'batter_ops', 'pitcher_opp_ops'
    ]
    target_col = 'home_win_flag'
    
    df_clean = df[feature_cols + [target_col]].dropna()
    
    return df_clean[feature_cols], df_clean[target_col]

def create_and_save_model():
    # 1. データ取得
    try:
        raw_df = fetch_real_data()
    except Exception as e:
        print(f"❌ データ取得エラー: {e}")
        return

    # 2. 前処理
    X, y = preprocess_data(raw_df)
    print(f"📊 学習データ数: {len(X)} 件")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. モデル学習 (軽量設定)
    print("🧠 AIモデルを学習中... (軽量設定)")
    # n_estimators(木の数)とmax_depth(深さ)を減らしてファイルサイズを小さくする
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 学習完了! テストデータでの正解率: {acc:.1%}")
    
    # 4. 保存
    # 圧縮レベルを指定して保存
    filename = 'baseball_model.pkl'
    joblib.dump(clf, filename, compress=3) 
    print(f"💾 モデルを圧縮して '{filename}' として保存しました。")

if __name__ == "__main__":
    create_and_save_model()