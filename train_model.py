import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pybaseball import statcast
import warnings
import os

# 警告を無視
warnings.filterwarnings('ignore')

def fetch_real_data(start_dt, end_dt):
    print(f"⚾ MLB公式データ(Statcast)を取得中... (期間: {start_dt} - {end_dt})")
    # データ量を削減するため、期間を指定して取得
    df = statcast(start_dt=start_dt, end_dt=end_dt)
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

    try:
        batter_ops = df.groupby('batter').apply(calculate_ops_simple).to_dict()
        pitcher_ops = df.groupby('pitcher').apply(calculate_ops_simple).to_dict()
        df['batter_ops'] = df['batter'].map(batter_ops).fillna(0.720)
        df['pitcher_opp_ops'] = df['pitcher'].map(pitcher_ops).fillna(0.720)
    except:
        df['batter_ops'] = 0.720
        df['pitcher_opp_ops'] = 0.720

    feature_cols = [
        'score_diff', 'inning', 'is_top', 'outs_when_up', 
        'on_1b', 'on_2b', 'on_3b',
        'batter_ops', 'pitcher_opp_ops'
    ]
    target_col = 'home_win_flag'
    
    df_clean = df[feature_cols + [target_col]].dropna()
    
    return df_clean[feature_cols], df_clean[target_col]

def save_model_split(model, filename, chunk_size=20 * 1024 * 1024): 
    """ 
    モデルを保存し、指定サイズ(デフォルト20MB)を超えたら分割する 
    """
    temp_name = f"temp_model.pkl"
    print(f"💾 モデルを一時保存中...")
    joblib.dump(model, temp_name, compress=3)
    
    file_size = os.path.getsize(temp_name)
    print(f"📦 モデルサイズ: {file_size / (1024*1024):.2f} MB")

    # ディレクトリが存在しない場合は作成
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 フォルダ作成: {directory}")

    # 分割不要な場合 (chunk_size以下)
    if file_size <= chunk_size:
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_name, filename)
        print(f"🎉 分割不要です。 '{filename}' として保存しました。")
        return

    # 分割処理
    print(f"✂️ サイズが大きいので {chunk_size / (1024*1024):.0f}MB ごとに分割します...")
    part_num = 0
    with open(temp_name, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_name = f"{filename}.part{part_num}"
            with open(part_name, "wb") as part_file:
                part_file.write(chunk)
            print(f"  -> {part_name} 保存完了")
            part_num += 1
            
    os.remove(temp_name) # 元ファイルは削除
    print("✅ 分割保存完了！Gitにはこれらの .part ファイルをアップロードしてください。")

def create_and_save_model():
    # 取得期間の設定
    start_dt = '2025-01-01'
    end_dt = '2025-12-31'

    try:
        raw_df = fetch_real_data(start_dt, end_dt)
    except Exception as e:
        print(f"❌ データ取得エラー: {e}")
        return

    X, y = preprocess_data(raw_df)
    
    if len(X) == 0:
        print("❌ データが空でした。期間を変更してください。")
        return

    print(f"📊 学習データ数: {len(X)} 件")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 AIモデルを学習中... (max_depth=30)")
    clf = RandomForestClassifier(n_estimators=50, max_depth=30, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 学習完了! テストデータでの正解率: {acc:.1%}")
    
    # 保存先の設定: baseball_modelフォルダの中に日付付きで保存
    save_folder = 'baseball_model'
    save_filename = os.path.join(save_folder, f'baseball_model({start_dt}ー{end_dt}).pkl')
    
    # 分割保存ロジックを使用
    save_model_split(clf, save_filename)

if __name__ == "__main__":
    create_and_save_model()