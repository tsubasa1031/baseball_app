import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pybaseball import statcast
import warnings
import os
import datetime
import gc # ガベージコレクション(メモリ解放)用
import calendar # カレンダー用

# 警告を無視
warnings.filterwarnings('ignore')

# --- 設定 ---
START_YEAR = 2024
END_YEAR = 2025
# メモリ不足になる場合はここを小さくしてください（例: 0.5 = 50%に間引く）
SAMPLE_FRAC = 1.0 

def fetch_long_term_data(start_year, end_year):
    """
    指定された年数のデータを1ヶ月ずつ取得して結合する
    """
    all_dfs = []
    
    # 必要なカラムだけに絞ることでメモリを節約
    cols_to_keep = [
        'game_pk', 'game_date', 'events', 
        'home_score', 'away_score', 
        'inning', 'inning_topbot', 'outs_when_up', 
        'on_1b', 'on_2b', 'on_3b',
        'batter', 'pitcher'
    ]

    print(f"⚾ {start_year}年から{end_year}年までのデータを取得します...")

    for year in range(start_year, end_year + 1):
        # レギュラーシーズンはおおよそ4月から10月 (3月〜11月までカバー)
        for month in range(3, 12): 
            # 日付範囲の設定
            start_dt = f"{year}-{month:02d}-01"
            # 月末の計算
            last_day = calendar.monthrange(year, month)[1]
            end_dt = f"{year}-{month:02d}-{last_day}"

            print(f"   📥 取得中: {start_dt} ～ {end_dt} ... ", end="")
            
            try:
                df_chunk = statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
                
                if df_chunk is not None and not df_chunk.empty:
                    # 必要な列のみ抽出してメモリ削減
                    # 存在しない列がある場合のエラー回避
                    available_cols = [c for c in cols_to_keep if c in df_chunk.columns]
                    df_small = df_chunk[available_cols].copy()
                    
                    # 型の最適化 (さらにメモリ削減)
                    for col in ['home_score', 'away_score', 'inning', 'outs_when_up', 'batter', 'pitcher']:
                        if col in df_small.columns:
                            df_small[col] = pd.to_numeric(df_small[col], errors='coerce').fillna(0).astype('int32')
                    
                    all_dfs.append(df_small)
                    print(f"OK ({len(df_small)}球)")
                else:
                    print("データなし")
                
                # メモリ解放 (取得成功時のみ実行)
                del df_chunk
                gc.collect()

            except Exception as e:
                print(f"スキップ ({e})")
                # 取得失敗時は df_chunk が存在しない可能性があるため del は行わない

    if not all_dfs:
        return pd.DataFrame()

    print("🔄 データを結合中...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # 指定があればサンプリング（間引き）
    if SAMPLE_FRAC < 1.0:
        print(f"✂️ データ量を {SAMPLE_FRAC*100}% に削減します...")
        full_df = full_df.sample(frac=SAMPLE_FRAC, random_state=42)
        
    print(f"✅ 合計データ数: {len(full_df)} 球")
    return full_df

def calculate_ops_fast(df):
    """
    高速化されたOPS計算ロジック（groupby + sum を使用）
    """
    print("   打者・投手の成績(OPS)を集計中 (高速版)...")
    
    # イベントを数値化
    # 打数に含まれるイベント
    ab_events = ['single', 'double', 'triple', 'home_run', 'field_out', 'strikeout', 'force_out', 'grounded_into_double_play']
    # ヒット
    hit_events = ['single', 'double', 'triple', 'home_run']
    
    df['is_ab'] = df['events'].isin(ab_events).astype(int)
    df['is_hit'] = df['events'].isin(hit_events).astype(int)
    df['is_single'] = (df['events'] == 'single').astype(int)
    df['is_double'] = (df['events'] == 'double').astype(int)
    df['is_triple'] = (df['events'] == 'triple').astype(int)
    df['is_hr'] = (df['events'] == 'home_run').astype(int)
    df['is_walk'] = df['events'].isin(['walk', 'hit_by_pitch', 'intent_walk']).astype(int)
    df['is_sf'] = (df['events'] == 'sac_fly').astype(int)
    
    # 塁打計算
    df['total_bases'] = df['is_single'] + (df['is_double']*2) + (df['is_triple']*3) + (df['is_hr']*4)

    # 集計用の関数
    def get_stats(group_col):
        stats = df.groupby(group_col)[['is_ab', 'is_hit', 'total_bases', 'is_walk', 'is_sf']].sum()
        # 出塁率 (OBP)
        obp_denom = stats['is_ab'] + stats['is_walk'] + stats['is_sf']
        obp = (stats['is_hit'] + stats['is_walk']) / obp_denom
        # 長打率 (SLG)
        slg_denom = stats['is_ab']
        slg = stats['total_bases'] / slg_denom
        
        # ゼロ除算対策
        obp = obp.fillna(0.320) # 平均的な値
        slg = slg.fillna(0.400)
        
        return (obp + slg).to_dict()

    # マッピング
    batter_ops_map = get_stats('batter')
    pitcher_ops_map = get_stats('pitcher')
    
    df['batter_ops'] = df['batter'].map(batter_ops_map).fillna(0.720)
    df['pitcher_opp_ops'] = df['pitcher'].map(pitcher_ops_map).fillna(0.720)
    
    # 不要になった一時列を削除
    drop_cols = ['is_ab', 'is_hit', 'is_single', 'is_double', 'is_triple', 'is_hr', 'is_walk', 'is_sf', 'total_bases']
    df.drop(columns=drop_cols, inplace=True)
    
    return df

def preprocess_data(df):
    print("⚙️ 前処理を実行中...")
    
    # 勝敗フラグ作成
    game_results = df.groupby('game_pk').agg({
        'home_score': 'max',
        'away_score': 'max'
    }).reset_index()
    
    game_results['home_win_flag'] = (game_results['home_score'] > game_results['away_score']).astype(int)
    df = df.merge(game_results[['game_pk', 'home_win_flag']], on='game_pk', how='left')
    
    # 特徴量作成
    df['score_diff'] = df['home_score'] - df['away_score']
    df['is_top'] = (df['inning_topbot'] == 'Top').astype(int)
    df['on_1b'] = df['on_1b'].notnull().astype(int)
    df['on_2b'] = df['on_2b'].notnull().astype(int)
    df['on_3b'] = df['on_3b'].notnull().astype(int)

    # OPS計算 (高速版)
    df = calculate_ops_fast(df)

    feature_cols = [
        'score_diff', 'inning', 'is_top', 'outs_when_up', 
        'on_1b', 'on_2b', 'on_3b',
        'batter_ops', 'pitcher_opp_ops'
    ]
    target_col = 'home_win_flag'
    
    df_clean = df[feature_cols + [target_col]].dropna()
    return df_clean[feature_cols], df_clean[target_col]

def save_model_split(model, filename, chunk_size=20 * 1024 * 1024): 
    temp_name = f"temp_model.pkl"
    print(f"💾 モデルを一時保存中...")
    joblib.dump(model, temp_name, compress=3)
    
    file_size = os.path.getsize(temp_name)
    print(f"📦 モデルサイズ: {file_size / (1024*1024):.2f} MB")

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if file_size <= chunk_size:
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_name, filename)
        print(f"🎉 分割不要です。 '{filename}' として保存しました。")
        return

    print(f"✂️ {chunk_size / (1024*1024):.0f}MB ごとに分割します...")
    part_num = 0
    with open(temp_name, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            part_name = f"{filename}.part{part_num}"
            with open(part_name, "wb") as part_file:
                part_file.write(chunk)
            print(f"  -> {part_name} 保存完了")
            part_num += 1
    os.remove(temp_name)
    print("✅ 分割保存完了！")

def create_and_save_model():
    # 1. 長期間データの取得
    try:
        raw_df = fetch_long_term_data(START_YEAR, END_YEAR)
    except Exception as e:
        print(f"❌ データ取得エラー: {e}")
        return

    if raw_df.empty:
        print("❌ データがありませんでした。")
        return

    # 2. 前処理
    X, y = preprocess_data(raw_df)
    print(f"📊 最終学習データ数: {len(X)} 件")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. モデル学習
    print(f"🧠 AIモデルを学習中... (n=100, depth=30)")
    # データが多いので、少し深さを制限しつつ木を増やす
    clf = RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 学習完了! テストデータでの正解率: {acc:.1%}")
    
    # 4. 保存
    save_folder = 'baseball_model'
    save_filename = os.path.join(save_folder, f'baseball_model({START_YEAR}ー{END_YEAR}).pkl')
    save_model_split(clf, save_filename)

if __name__ == "__main__":
    create_and_save_model()