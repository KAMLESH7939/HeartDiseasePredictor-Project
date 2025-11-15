import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px


# =========================================================
# 📌 Load Background Image as Base64
# =========================================================
def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_base64 = get_base64_of_image("gradientt.jpg")


# =========================================================
# 📌 Inject Custom Premium NEON UI CSS
# =========================================================
st.markdown(
    f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
}}

/* ===============================
   BACKGROUND
================================ */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}

html, body {{
    background-color: transparent !important;
}}

/* ===============================
   MAIN CONTENT CONTAINER
================================ */
.block-container {{
    max-width: 850px;
    margin: auto;
    padding-top: 40px;
}}

/* ===============================
   INPUTS — Glassmorphism + Neon Hover
================================ */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div {{
    background: rgba(0,0,0,0.45) !important;
    color: #fff !important;
    border-radius: 12px !important;
    padding: 10px;
    border: 1px solid rgba(255, 0, 255, 0.3);
    transition: 0.3s ease;
}}

.stTextInput>div>div>input:hover,
.stNumberInput>div>div>input:hover,
.stSelectbox>div>div>div:hover {{
    border: 1px solid #ff00ff;
    box-shadow: 0 0 10px #ff00ff;
    transform: translateY(-2px);
}}

/* ===============================
   LABELS
================================ */
label {{
    font-size: 18px !important;
    font-weight: 500 !important;
    color: #ffd9ff !important;
}}

/* ===============================
   TABS — Neon Bottom Glow
================================ */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(8px);
    border-radius: 12px;
}}

.stTabs [data-baseweb="tab"] {{
    font-size: 17px !important;
    color: #dbaaff !important;
    font-weight: 500;
    padding: 10px 20px !important;
    transition: 0.3s ease;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: #fff !important;
    font-weight: 600;
    border-bottom: 3px solid #ff00ff !important;
    text-shadow: 0 0 8px #ff00ff;
}}

/* ===============================
   NEON BUTTONS — Animated
================================ */
.stButton>button {{
    background: linear-gradient(90deg, #ff00ff, #ff44cc) !important;
    color: white !important;
    padding: 10px 25px;
    font-size: 18px;
    border-radius: 12px;
    border: none;
    box-shadow: 0 0 12px #ff00ff;
    cursor: pointer;
    transition: 0.3s ease-in-out;
    font-weight: 600;
}}

.stButton>button:hover {{
    transform: scale(1.07);
    box-shadow: 0 0 20px #ff00ff, 0 0 30px #ff44cc;
}}

/* ===============================
   RESULT CARDS — Animated Gradient Border
================================ */
.result-card {{
    background: rgba(20, 20, 35, 0.55);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    margin-top: 25px;

    border: 2px solid transparent;
    background-image:
        linear-gradient(rgba(20,20,35,0.65), rgba(20,20,35,0.65)),
        linear-gradient(45deg, #ff00ff, #7700ff, #ff0080);

    background-clip: padding-box, border-box;

    box-shadow: 0 0 15px rgba(255, 0, 255, 0.25);
    transition: 0.35s ease;
}}

.result-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 0 25px #ff00ff;
}}

.result-title {{
    font-size: 24px;
    font-weight: 600;
    color: #ffb3ff;
    margin-bottom: 10px;
}}

.result-text {{
    font-size: 18px;
    color: white;
    opacity: 0.9;
}}

/* ===============================
   SIDEBAR — Futuristic Neon Panel
================================ */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.4) !important;
    border-right: 2px solid rgba(255,0,255,0.25);
    backdrop-filter: blur(12px);
}}

[data-testid="stSidebar"] * {{
    color: #ffccff !important;
}}

[data-testid="stSidebar"] a:hover {{
    text-shadow: 0 0 8px #ff00ff;
    color: white !important;
}}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #fff !important;
    font-weight: 600 !important;
    text-shadow: 0 0 8px #ff00ff;
}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# 📌 CSV Downloader
# =========================================================
def get_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predicted.csv">Download CSV File</a>'


# =========================================================
# 📌 STREAMLIT APP
# =========================================================
st.title("Cardiac Disease Detection Model")

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])


# =========================================================
# 🩺 TAB 1 — SINGLE PREDICTION
# =========================================================
with tab1:

    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    rbp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, max_value=300)
    chol = st.number_input("Cholesterol (in mg/dl)", min_value=0, max_value=1000)
    fbs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    maxhr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, format="%.1f")
    st_slope = st.selectbox("Slope of ST segment", ["Upsloping", "Flat", "Downsloping"])

    # Encode
    sex = 0 if sex == "Male" else 1
    cp_dict = {"Typical Angina": 3, "Atypical Angina": 0, "Non-anginal Pain": 1, "Asymptomatic": 2}
    ecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    fbs = 0 if fbs == "<= 120 mg/dl" else 1
    exang = 1 if exang == "Yes" else 0

    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [cp_dict[cp]],
        'RestingBP': [rbp], 'Cholesterol': [chol], 'FastingBS': [fbs],
        'RestingECG': [ecg_dict[ecg]], 'MaxHR': [maxhr],
        'ExerciseAngina': [exang], 'Oldpeak': [oldpeak],
        'ST_Slope': [slope_dict[st_slope]]
    })

    algos = ["Logistic Regression", "Decision Tree", "Random Forest", "MLP", "CNN"]
    files = [
        "logistic_regression_model.pkl",
        "decision_tree_model.pkl",
        "random_forest_model.pkl",
        "mlp_model.keras",
        "cnn_1d_model.keras"
    ]

    def predict_all(data):
        results = []
        for mdl in files:
            if mdl.endswith(".pkl"):
                model = pickle.load(open(mdl, "rb"))
                pred = model.predict(data)
            else:
                model = tf.keras.models.load_model(mdl)
                pred = model.predict(data.values.reshape(1, -1, 1) if "cnn" in mdl else data.values)
                pred = (pred > 0.5).astype(int)
            results.append(pred)
        return results

    if st.button("Predict"):
        st.subheader("Prediction Results")
        st.markdown("---")

        preds = predict_all(input_data)

        for i in range(len(preds)):
            diagnosis = "No Heart Disease" if preds[i][0] == 0 else "Heart Disease Detected"

            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-title">{algos[i]}</div>
                  <div class="result-text">{diagnosis}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# 📁 TAB 2 — BULK CSV PREDICTION
# =========================================================
with tab2:

    st.subheader("Upload CSV for Bulk Prediction")
    uploaded = st.file_uploader("Choose CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        model = pickle.load(open("logistic_regression_model.pkl", "rb"))

        maps = {
            "Sex": {"M": 0, "F": 1, "Male": 0, "Female": 1},
            "ChestPainType": {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3},
            "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
            "ExerciseAngina": {"Y": 1, "N": 0, "Yes": 1, "No": 0},
            "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}
        }

        for col, mapping in maps.items():
            if df[col].dtype == object:
                df[col] = df[col].map(mapping)

        df = df.astype(float)
        df["Prediction"] = df.apply(lambda r: model.predict([r.values])[0], axis=1)

        st.write(df)
        st.markdown(get_downloader_html(df), unsafe_allow_html=True)


# =========================================================
# 📊 TAB 3 — MODEL INFORMATION
# =========================================================
with tab3:
    stats = {
        'Logistic Regression': 85.86,
        'Decision Tree': 80.97,
        'Random Forest': 88.04,
        'MLP': 75.54,
        'CNN': 86.95
    }

    df = pd.DataFrame({"Models": list(stats.keys()), "Accuracy": list(stats.values())})

    fig = px.bar(df, x='Models', y='Accuracy', color='Accuracy',
                 text='Accuracy', title='Model Accuracy Comparison')

    st.plotly_chart(fig)
