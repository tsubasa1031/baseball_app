import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import io

# 機械学習ライブラリの読み込み
try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    ml_available = True
except ImportError:
    ml_available = False

st.set_page_config(page_title="野球勝率シミュレーター", page_icon="⚾", layout="centered")

# --- 設定 ---
EXTERNAL_MODEL_URL = "" 

# --- 状態管理 ---
default_state = {
    "inning": 9,
    "top_bot": "裏",
    "score_away": 4,
    "score_home": 3,
    "outs": 0,
    "balls": 0,
    "strikes": 0,
    "runner_1": False,
    "runner_2": False,
    "runner_3": False,
}

for key, val in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- モデル読み込みロジック ---
def load_split_model(base_filepath):
    part_files = []
    i = 0
    while True:
        part_name = f"{base_filepath}.part{i}"
        if os.path.exists(part_name):
            part_files.append(part_name)
            i += 1
        else:
            break
            
    if not part_files:
        return None

    combined_data = bytearray()
    for part in part_files:
        with open(part, "rb") as f:
            combined_data.extend(f.read())
            
    try:
        return joblib.load(io.BytesIO(combined_data))
    except Exception as e:
        return None

@st.cache_resource
def load_or_train_model():
    if not ml_available:
        return None, "unavailable", []

    # 探索するモデルのパス候補
    candidates = ['baseball_model.pkl']
    
    model_dir = 'baseball_model'
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        # .pkl ファイルを探す
        pkl_candidates = [os.path.join(model_dir, f) for f in files if f.endswith('.pkl')]
        # .pkl.part0 ファイル（分割ファイル）を探す
        part_candidates = [os.path.join(model_dir, f.replace('.part0', '')) for f in files if f.endswith('.pkl.part0')]
        
        candidates.extend(pkl_candidates)
        candidates.extend(part_candidates)

    # 候補を名前順（降順＝新しい日付順）にソート
    candidates.sort(reverse=True)

    load_errors = [] # エラーログ用

    for model_path in candidates:
        # 1. 通常ファイル
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path), f"loaded ({os.path.basename(model_path)})", []
            except Exception as e:
                load_errors.append(f"エラー ({os.path.basename(model_path)}): {str(e)}")
                continue

        # 2. 分割ファイル
        try:
            split_model = load_split_model(model_path)
            if split_model:
                return split_model, f"loaded split ({os.path.basename(model_path)})", []
        except Exception as e:
             load_errors.append(f"分割エラー ({os.path.basename(model_path)}): {str(e)}")

    # 3. 全て失敗した場合 -> デモ学習
    n_samples = 3000
    X = [] 
    y = []
    np.random.seed(42)
    for _ in range(n_samples):
        inn = np.random.randint(1, 10)
        is_top = np.random.randint(0, 2)
        diff = np.random.randint(-6, 7)
        out = np.random.randint(0, 3)
        r1 = np.random.randint(0, 2)
        r2 = np.random.randint(0, 2)
        r3 = np.random.randint(0, 2)
        prob = 0.5 + (diff * 0.1)
        if inn >= 7: prob += (diff * 0.05)
        runners_score = r1 + r2*1.5 + r3*2
        if is_top == 1: 
            prob -= (runners_score * 0.05)
            prob += (out * 0.03)
        else:
            prob += (runners_score * 0.05)
            prob -= (out * 0.03)
        prob = max(0.05, min(0.95, prob))
        win = 1 if np.random.rand() < prob else 0
        X.append([diff, inn, is_top, out, r1, r2, r3])
        y.append(win)
        
    clf = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    clf.fit(X, y)
    
    return clf, "trained (demo)", load_errors

# モデル取得
ml_model, model_source, error_logs = load_or_train_model()

# エラーがあった場合の表示（デバッグ用）
if error_logs:
    st.error("⚠️ モデルファイルの読み込みに失敗しました")
    with st.expander("エラー詳細を確認する"):
        for err in error_logs:
            st.code(err)

# --- ロジック関数 ---
def reset_all_situation():
    st.session_state.balls = 0
    st.session_state.strikes = 0
    st.session_state.runner_1 = False
    st.session_state.runner_2 = False
    st.session_state.runner_3 = False

def add_ball():
    if st.session_state.balls < 3:
        st.session_state.balls += 1
    else:
        st.session_state.balls = 0
        st.session_state.strikes = 0
        if not st.session_state.runner_1:
            st.session_state.runner_1 = True
        elif not st.session_state.runner_2:
            st.session_state.runner_2 = True
        elif not st.session_state.runner_3:
            st.session_state.runner_3 = True

def add_strike():
    if st.session_state.strikes < 2:
        st.session_state.strikes += 1
    else:
        st.session_state.strikes = 0
        st.session_state.balls = 0
        add_out()

def add_out():
    if st.session_state.outs < 2:
        st.session_state.outs += 1
        st.session_state.balls = 0
        st.session_state.strikes = 0
    else:
        st.session_state.outs = 0
        st.session_state.balls = 0
        st.session_state.strikes = 0
        if st.session_state.top_bot == "表":
            st.session_state.top_bot = "裏"
        else:
            st.session_state.top_bot = "表"
            st.session_state.inning += 1
        st.session_state.runner_1 = False
        st.session_state.runner_2 = False
        st.session_state.runner_3 = False

def calculate_win_prob_simple():
    s = st.session_state
    score_diff = s.score_home - s.score_away
    base_prob = 50 + (score_diff * 10)
    runner_count = sum([s.runner_1, s.runner_2, s.runner_3])
    runner_bonus = runner_count * 5
    count_advantage = (s.balls * 1) - (s.strikes * 2)
    out_penalty = s.outs * 4
    if s.top_bot == "表":
        current_prob = base_prob - runner_bonus - count_advantage + out_penalty
    else:
        current_prob = base_prob + runner_bonus + count_advantage - out_penalty
    urgency = 1 + (s.inning / 8)
    final_prob = 50 + ((current_prob - 50) * urgency)
    return max(0.1, min(99.9, final_prob))

def calculate_win_prob_ml():
    if ml_model is None:
        return calculate_win_prob_simple()

    s = st.session_state
    score_diff = s.score_home - s.score_away
    is_top_val = 1 if s.top_bot == "表" else 0
    
    # 基本の7特徴量
    input_data = [
        score_diff,
        s.inning,
        is_top_val,
        s.outs,
        int(s.runner_1),
        int(s.runner_2),
        int(s.runner_3)
    ]
    
    # モデルがOPS等の追加特徴量を求めている場合の対応
    if hasattr(ml_model, "n_features_in_") and ml_model.n_features_in_ > len(input_data):
        missing_cols = ml_model.n_features_in_ - len(input_data)
        # 不足分を平均値(0.720)で埋める
        input_data.extend([0.720] * missing_cols)
        
    try:
        prob = ml_model.predict_proba([input_data])[0][1]
        return prob * 100
    except Exception:
        return calculate_win_prob_simple()

win_prob = calculate_win_prob_ml()
away_prob = 100 - win_prob

# --- CSSスタイル ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; }
    .scoreboard-table { width: 100%; border-collapse: collapse; background-color: #0f172a; color: white; font-family: 'Courier New', monospace; border-radius: 8px; overflow: hidden; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .scoreboard-table th, .scoreboard-table td { border: 1px solid #334155; padding: 0.5rem; text-align: center; width: 8%; }
    .scoreboard-table th { background-color: #1e293b; font-weight: bold; color: #94a3b8; }
    .team-name { text-align: left !important; width: 20% !important; font-weight: bold; padding-left: 1rem !important; }
    .score-total { font-weight: 900; font-size: 1.2rem; background-color: #334155; color: #fbbf24; }
    .control-label { font-size: 0.8rem; font-weight: bold; color: #64748b; text-align: center; }
    .field-container { position: relative; width: 100%; max-width: 400px; aspect-ratio: 1 / 0.8; margin: 0 auto; background-color: #15803d; border-radius: 8px 8px 100px 100px; border: 4px solid #14532d; overflow: hidden; box-shadow: inset 0 0 20px rgba(0,0,0,0.3); }
    .mound { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 12%; height: 15%; background-color: #d97706; border-radius: 50%; border: 2px solid #fff; opacity: 0.8; }
    .base { position: absolute; width: 8%; height: 10%; background-color: white; transform: rotate(45deg); box-shadow: 2px 2px 4px rgba(0,0,0,0.3); transition: background-color 0.2s; z-index: 10; }
    .base.active { background-color: #fbbf24; box-shadow: 0 0 10px #fbbf24; }
    .base-1 { top: 50%; right: 20%; transform: translate(50%, -50%) rotate(45deg); }
    .base-2 { top: 20%; left: 50%; transform: translate(-50%, -50%) rotate(45deg); }
    .base-3 { top: 50%; left: 20%; transform: translate(-50%, -50%) rotate(45deg); }
    .base-home { position: absolute; bottom: 10%; left: 50%; width: 8%; height: 5%; background-color: white; clip-path: polygon(0 0, 50% 100%, 100% 0); transform: translate(-50%, 0); z-index: 10; }
    .foul-line { position: absolute; bottom: 10%; left: 50%; width: 60%; height: 2px; background-color: rgba(255,255,255,0.5); transform-origin: left center; }
    .line-left { transform: rotate(-45deg); }
    .line-right { transform: rotate(-135deg); }
    .count-board { position: absolute; top: 5%; left: 5%; background-color: rgba(0,0,0,0.7); padding: 5px 10px; border-radius: 6px; color: white; font-family: sans-serif; font-size: 0.8rem; font-weight: bold; z-index: 20; border: 1px solid #475569; }
    .lamp-row { display: flex; align-items: center; margin-bottom: 2px; }
    .lamp-label { width: 15px; text-align: center; margin-right: 5px; font-size: 0.7rem; }
    .lamp { width: 8px; height: 8px; border-radius: 50%; background-color: #334155; margin-right: 3px; border: 1px solid #64748b; }
    .lamp.b-active { background-color: #22c55e; box-shadow: 0 0 5px #22c55e; }
    .lamp.s-active { background-color: #eab308; box-shadow: 0 0 5px #eab308; }
    .lamp.o-active { background-color: #ef4444; box-shadow: 0 0 5px #ef4444; }
    .win-prob-wrapper { margin-top: 1rem; margin-bottom: 2rem; }
    .win-prob-bar { height: 20px; width: 100%; background: #e2e8f0; border-radius: 10px; overflow: hidden; display: flex; }
    .bar-away { background: #3b82f6; display: flex; align-items: center; padding-left: 8px; color: white; font-size: 10px; font-weight: bold; }
    .bar-home { background: #ef4444; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; color: white; font-size: 10px; font-weight: bold; }
    .control-card { background: white; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
</style>
""", unsafe_allow_html=True)

innings_html = "".join([f"<th>{i}</th>" for i in range(1, 10)])
inning_cells = "<td></td>" * 9
st.markdown(f"""
<table class="scoreboard-table">
    <thead><tr><th class="team-name">TEAM</th>{innings_html}<th class="score-total">R</th><th>H</th><th>E</th></tr></thead>
    <tbody>
        <tr><td class="team-name" style="color:#60a5fa;">VISITOR</td>{inning_cells}<td class="score-total">{st.session_state.score_away}</td><td>-</td><td>-</td></tr>
        <tr><td class="team-name" style="color:#f87171;">HOME</td>{inning_cells}<td class="score-total">{st.session_state.score_home}</td><td>-</td><td>-</td></tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 0.8, 1])
with c1:
    st.markdown('<div class="control-label" style="color:#3b82f6;">VISITOR</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([1, 1.5, 1])
    if sc1.button("－", key="aw_m"): st.session_state.score_away = max(0, st.session_state.score_away - 1); st.rerun()
    with sc2: st.markdown(f"<div style='text-align:center; font-size:2rem; font-weight:bold; line-height:1;'>{st.session_state.score_away}</div>", unsafe_allow_html=True)
    if sc3.button("＋", key="aw_p"): st.session_state.score_away += 1; st.rerun()
with c2:
    st.markdown('<div class="control-label">INNING</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns([1, 2, 1])
    if ic1.button("◀", key="inn_m"):
        if st.session_state.inning > 1: st.session_state.inning -= 1; st.rerun()
    with ic2:
        if st.button(f"{st.session_state.inning}{st.session_state.top_bot}", key="tb_toggle", use_container_width=True):
            st.session_state.top_bot = "裏" if st.session_state.top_bot == "表" else "表"; st.rerun()
    if ic3.button("▶", key="inn_p"): st.session_state.inning += 1; st.rerun()
with c3:
    st.markdown('<div class="control-label" style="color:#ef4444;">HOME</div>', unsafe_allow_html=True)
    hc1, hc2, hc3 = st.columns([1, 1.5, 1])
    if hc1.button("－", key="hm_m"): st.session_state.score_home = max(0, st.session_state.score_home - 1); st.rerun()
    with hc2: st.markdown(f"<div style='text-align:center; font-size:2rem; font-weight:bold; line-height:1;'>{st.session_state.score_home}</div>", unsafe_allow_html=True)
    if hc3.button("＋", key="hm_p"): st.session_state.score_home += 1; st.rerun()

st.markdown(f"""
<div class="win-prob-wrapper">
    <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px; font-weight:bold;">
        <span style="color:#3b82f6;">Visitor: {away_prob:.1f}%</span><span style="color:#ef4444;">Home: {win_prob:.1f}%</span>
    </div>
    <div class="win-prob-bar">
        <div class="bar-away" style="width: {away_prob}%;">AWAY</div><div class="bar-home" style="width: {win_prob}%;">HOME</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_field, col_ctrl = st.columns([1.3, 1])
with col_field:
    c1b, c2b, c3b = ("active" if st.session_state[k] else "" for k in ["runner_1", "runner_2", "runner_3"])
    b_html = "".join([f'<div class="lamp {"b-active" if i < st.session_state.balls else ""}"></div>' for i in range(3)])
    s_html = "".join([f'<div class="lamp {"s-active" if i < st.session_state.strikes else ""}"></div>' for i in range(2)])
    o_html = "".join([f'<div class="lamp {"o-active" if i < st.session_state.outs else ""}"></div>' for i in range(2)])
    st.markdown(f"""<div class="field-container"><div class="count-board"><div class="lamp-row"><div class="lamp-label" style="color:#22c55e">B</div>{b_html}</div><div class="lamp-row"><div class="lamp-label" style="color:#eab308">S</div>{s_html}</div><div class="lamp-row"><div class="lamp-label" style="color:#ef4444">O</div>{o_html}</div></div><div class="foul-line line-left"></div><div class="foul-line line-right"></div><div class="mound"></div><div class="base base-1 {c1b}"></div><div class="base base-2 {c2b}"></div><div class="base base-3 {c3b}"></div><div class="base-home"></div></div>""", unsafe_allow_html=True)

with col_ctrl:
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.caption("🏃 ランナー")
    r2c = st.columns([1,1,1]); 
    if r2c[1].button("2塁", type="primary" if st.session_state.runner_2 else "secondary", use_container_width=True): st.session_state.runner_2 = not st.session_state.runner_2; st.rerun()
    r13c = st.columns([1,1,1]); 
    if r13c[0].button("3塁", type="primary" if st.session_state.runner_3 else "secondary", use_container_width=True): st.session_state.runner_3 = not st.session_state.runner_3; st.rerun()
    if r13c[2].button("1塁", type="primary" if st.session_state.runner_1 else "secondary", use_container_width=True): st.session_state.runner_1 = not st.session_state.runner_1; st.rerun()
    st.divider(); st.caption("⚾ カウント")
    cc = st.columns(3)
    with cc[0]: st.markdown(f"<div style='text-align:center;font-weight:bold;color:#22c55e;font-size:0.8rem'>B {st.session_state.balls}</div>", unsafe_allow_html=True); st.button("＋B", use_container_width=True, on_click=add_ball)
    with cc[1]: st.markdown(f"<div style='text-align:center;font-weight:bold;color:#eab308;font-size:0.8rem'>S {st.session_state.strikes}</div>", unsafe_allow_html=True); st.button("＋S", use_container_width=True, on_click=add_strike)
    with cc[2]: st.markdown(f"<div style='text-align:center;font-weight:bold;color:#ef4444;font-size:0.8rem'>O {st.session_state.outs}</div>", unsafe_allow_html=True); st.button("＋O", use_container_width=True, on_click=add_out)
    if st.button("状況リセット", use_container_width=True, type="secondary"): reset_all_situation(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("詳細設定・モデル情報"):
    if "split" in str(model_source): st.success(f"✅ 分割モデル: {model_source}")
    elif "loaded" in str(model_source): st.success(f"✅ 学習済みモデル: {model_source}")
    else: st.info("ℹ️ デモモード: 簡易モデルを使用中")
    if error_logs: st.warning(f"読込エラー ({len(error_logs)}件) が発生しました。")