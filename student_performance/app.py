"""Streamlit dashboard.  Run:  streamlit run app.py   (after python pipeline.py)"""
import os, joblib
import pandas as pd, streamlit as st
from pipeline import load_data, clean, TARGET

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")

DATA = st.sidebar.text_input("Dataset path", "data/student-mat.csv")
if not (os.path.exists("models/best_model.joblib") and os.path.exists(DATA)):
    st.error("Pehle `python pipeline.py` chalao aur dataset `data/` folder me rakho.")
    st.stop()

bundle = joblib.load("models/best_model.joblib")
df = clean(load_data(DATA))
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA", "🏆 Model Comparison"])

with tab1:
    st.caption(f"Model: **{bundle['best']}** | Final grade (G3) 0-20 scale me predict hota hai")
    c1, c2, c3 = st.columns(3)
    studytime = c1.select_slider("Study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", [1, 2, 3, 4], 2)
    absences = c2.slider("Absences (attendance)", 0, 75, 4)
    failures = c3.selectbox("Past failures", [0, 1, 2, 3])
    g1 = c1.slider("Previous grade G1", 0, 20, 10)
    g2 = c2.slider("Previous grade G2", 0, 20, 10)

    row = dict(bundle["defaults"])          # other numeric features -> median
    for c, opts in bundle["cats"].items():   # categorical -> most common value
        row[c] = df[c].mode()[0]
    row.update(studytime=studytime, absences=absences, failures=failures, G1=g1, G2=g2)
    pred = float(min(20, max(0, bundle["pipeline"].predict(pd.DataFrame([row])[bundle["features"]])[0])))

    st.metric("Predicted final grade (G3)", f"{pred:.1f} / 20")
    if pred < 10:
        st.error("⚠️ At-risk student: extra support/mentoring recommended.")
    elif pred < 14:
        st.warning("Average performance: monitor progress.")
    else:
        st.success("Good performance expected.")

with tab2:
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.subheader("G3 distribution"); c1.bar_chart(df[TARGET].value_counts().sort_index())
    c2.subheader("Avg G3 by study time"); c2.bar_chart(df.groupby("studytime")[TARGET].mean())
    st.subheader("Correlation with G3")
    st.bar_chart(df.select_dtypes("number").corr()[TARGET].drop(TARGET).sort_values())
    for f in ["eda_overview.png", "correlation_heatmap.png"]:
        if os.path.exists(f"outputs/{f}"): st.image(f"outputs/{f}")

with tab3:
    res = pd.read_csv("outputs/model_comparison.csv")
    st.dataframe(res, use_container_width=True)
    st.bar_chart(res.set_index("Model")[["RMSE"]])
    if os.path.exists("outputs/actual_vs_predicted.png"): st.image("outputs/actual_vs_predicted.png", width=450)
