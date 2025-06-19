import streamlit as st
import pandas as pd
from numbers3_predictor import main_with_improved_predictions

st.set_page_config(page_title="Numbers3予測AI", layout="centered")

# --- スタイル ---
st.markdown("""
<style>
.card {
    background-color: #f0f2f6;
    padding: 1em;
    margin-bottom: 1em;
    border-radius: 1em;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.card h3 {
    margin: 0.2em 0;
    color: #333;
}
.badge {
    display: inline-block;
    padding: 0.4em 0.7em;
    border-radius: 0.5em;
    font-size: 0.8em;
    color: white;
}
.badge.straight { background: #e63946; }
.badge.box { background: #f4a261; }
.badge.mini { background: #2a9d8f; }
.badge.miss { background: #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- タイトル ---
st.title("🎯 Numbers3 予測AI")

# --- 最新予測を読み込み ---
try:
    df = pd.read_csv("numbers3_predictions.csv")
    latest = df.sort_values("抽せん日", ascending=False).iloc[-1]

    st.markdown("## 📌 最新予測（" + latest["抽せん日"] + "）")

    for i in range(1, 6):
        numbers = latest[f"予測{i}"]
        confidence = latest[f"信頼度{i}"]
        source = latest.get(f"出力元{i}", "AI")

        # 等級予測色バッジ（仮で信頼度で分類）
        if confidence >= 0.94:
            grade = "ストレート"
            badge_class = "straight"
        elif confidence >= 0.92:
            grade = "ボックス"
            badge_class = "box"
        elif confidence >= 0.90:
            grade = "ミニ"
            badge_class = "mini"
        else:
            grade = "はずれ"
            badge_class = "miss"

        st.markdown(f"""
        <div class="card">
            <h3>🎱 予測{i}: <code>{numbers}</code></h3>
            <div class="badge {badge_class}">{grade}</div>
            <span>信頼度: <b>{confidence:.3f}</b>｜出力元: {source}</span>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.warning("まだ予測が実行されていません。")

# --- 実行ボタン ---
st.markdown("---")
if st.button("📈 予測を再実行する"):
    with st.spinner("予測中..."):
        main_with_improved_predictions()
    st.success("予測が更新されました！ページを再読み込みしてください。")
