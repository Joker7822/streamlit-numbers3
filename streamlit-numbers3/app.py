
import streamlit as st
import pandas as pd
from numbers3_predictor import main_with_improved_predictions

st.set_page_config(page_title="Numbers3予測AI", layout="centered")

st.title("🎯 Numbers3 予測AI")

if st.button("📈 予測を実行する"):
    st.info("予測中...")
    main_with_improved_predictions()
    st.success("予測完了！")

if st.checkbox("📄 最新予測を表示"):
    try:
        df = pd.read_csv("numbers3_predictions.csv")
        st.dataframe(df.tail(10), use_container_width=True)
    except Exception as e:
        st.error(f"読み込み失敗: {e}")
