import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px

# -------------------------
# Load background image as base64
# -------------------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

background_base64 = get_base64_image("gradientt.jpg")

# -------------------------
# Inject CSS
# -------------------------
st.markdown(
    f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
    color: #f1d9ff;
}}

[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}

.block-container {{
    max-width: 1000px !important;
    padding-top: 25px;
}}

/* ============================
   INPUT CARD (Option A - Full Width)
============================= */
.input-card {{
    position: relative;
    border-radius: 25px;
    padding: 22px;
    margin-bottom: 25px;
    width: 100%;

    background-image:
        linear-gradient(rgba(18,12,25,0.85), rgba(18,12,25,0.85)),
        linear-gradient(90deg, #ff00ff, #8b00ff, #ff0099);
    background-origin: border-box;
    background-clip: padding-box, border-box;

    border: 2px solid transparent;
    box-shadow: 0 8px 35px rgba(255,0,255,0.15),
                0 0 25px rgba(120,0,255,0.12);

    transition: 0.25s ease-in-out;
}}

.input-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 18px 55px rgba(255,0,255,0.25),
                0 0 45px rgba(120,0,255,0.20);
}}

.input-card label {{
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}}

.input-card .stNumberInput>div>div>input,
.input-card .stTextInput>div>div>input,
.input-card .stSelectbox>div>div>div {{
    background: rgba(0,0,0,0.55) !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,0,255,0.25) !important;
    padding: 12px !important;
    font-size: 17px !important;
}}

.input-card .stNumberInput>div>div>input:hover,
.input-card .stTextInput>div>div>input:hover,
.input-card .stSelectbox>div>div>div:hover {{
    border-color: #ff00ff !important;
    box-shadow: 0 0 12px rgba(255,0,255,0.2);
}}

.input-card input::placeholder {{
    color: rgba(255,255,255,0.6);
}}

/* ============================
   Center Predict Button (FULL WIDTH LAYOUT)
============================= */
.center-btn {{
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 15px;
    margin-bottom: 15px;
}}

.stButton>button {{
    background: linear-gradient(90deg, #ff00ff, #ff33cc, #ff66aa);
    border: none;
    color: white !important;
    padding: 14px 40px;
    font-size: 20px;
    font-weight: 700;
    border-radius: 14px;
    cursor: pointer;

    box-shadow: 0 8px 25px rgba(255,0,255,0.25),
                inset 0 0 10px rgba(255,255,255,0.15);

    transition: 0.2s ease-in-out;
}}

.stButton>button:hover {{
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0 18px 45px rgba(255,0,255,0.35),
                0 0 55px rgba(255,0,200,0.25);
}}

.stButton>button:active {{
    transform: scale(0.98);
}}

/* ============================
   RESULT CARDS - FULL WIDTH
============================= */
.result-card {{
    width: 100%;
    padding: 25px;
    margin-top: 20px;

    border-radius: 25px;
    background-image:
        linear-gradient(rgba(30,18,45,0.85), rgba(30,18,45,0.85)),
        linear-gradient(45deg, #ff00ff, #7a00ff, #ff0090);
    background-origin: border-box;
    background-clip: padding-box, border-box;

    border: 2px solid transparent;
    animation: neonPulse 3s ease-in-out infinite;

    transition: 0.2s ease-in-out;
}}

.result-card:hover {{
    transform: translateY(-10px);
    box-shadow: 0 25px 60px rgba(255,0,255,0.3),
                0 0 60px rgba(120,0,255,0.25);
}}

@keyframes neonPulse {{
    0% {{
        box-shadow: 0 0 12px rgba(255,0,255,0.25),
                    0 0 25px rgba(120,0,255,0.18);
    }}
    50% {{
        box-shadow: 0 0 30px rgba(255,0,255,0.5),
                    0 0 55px rgba(255,80,180,0.3);
    }}
    100% {{
        box-shadow: 0 0 12px rgba(255,0,255,0.25),
                    0 0 25px rgba(120,0,255,0.18);
    }}
}}

.result-title {{
    font-size: 22px;
    font-weight: 700;
    color: #ffe6ff;
}}

.result-text {{
    font-size: 18px;
    margin-top: 10px;
    color: #ffffff;
}}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Download CSV Helper
# -------------------------
def get_downloader_html(df):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predicted.csv">Download CSV</a>'

# -------------------------
# PAGE TITLE
# -------------------------
st.title("Cardiac Disease Detection Model")

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# -------------------------
# CARD INPUT FUNCTIONS
# -------------------------
def card_input_number(label, **kwargs):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    value = st.number_input(label, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)
    return value

def card_input_select(label, options):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    value = st.selectbox(label, options)
    st.markdown('</div>', unsafe_allow_html=True)
    return value

# -------------------------
# TAB 1 — PREDICT
# -------------------------
with tab1:

    age = card_input_number("Age", min_value=1, max_value=120, value=45)
    sex = card_input_select("Sex", ["Male", "Female"])
    cpt = card_input_select("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
    rbp = card_input_number("Resting Blood Pressure", min_value=0, max_value=300, value=120)
    chol = card_input_number("Cholesterol (mg/dl)", min_value=0, max_value=1000, value=200)
    fbs = card_input_select("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    ecg = card_input_select("Resting ECG", ["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
    mhr = card_input_number("Max Heart Rate Achieved", min_value=60, max_value=202, value=150)
    ang = card_input_select("Exercise Induced Angina", ["Yes","No"])
    old = card_input_number("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    slope = card_input_select("Slope of peak exercise ST segment", ["Upsloping","Flat","Downsloping"])

    # Encoding
    df = pd.DataFrame({
        "Age": [age],
        "Sex": [0 if sex=="Male" else 1],
        "ChestPainType": [{"Typical Angina":3,"Atypical Angina":0,"Non-anginal Pain":1,"Asymptomatic":2}[cpt]],
        "RestingBP": [rbp],
        "Cholesterol": [chol],
        "FastingBS": [0 if fbs=="<= 120 mg/dl" else 1],
        "RestingECG": [{"Normal":0,"ST-T wave abnormality":1,"Left ventricular hypertrophy":2}[ecg]],
        "MaxHR": [mhr],
        "ExerciseAngina": [1 if ang=="Yes" else 0],
        "Oldpeak": [old],
        "ST_Slope": [{"Upsloping":0,"Flat":1,"Downsloping":2}[slope]],
    })

    models = [
        ("Logistic Regression","logistic_regression_model.pkl"),
        ("Decision Tree","decision_tree_model.pkl"),
        ("Random Forest","random_forest_model.pkl"),
        ("MLP","mlp_model.keras"),
        ("CNN","cnn_1d_model.keras"),
    ]

    # CENTERED BUTTON
    st.markdown("<div class='center-btn'>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        for name, path in models:
            try:
                if path.endswith(".pkl"):
                    model = pickle.load(open(path,"rb"))
                    pred = int(model.predict(df)[0])
                else:
                    model = tf.keras.models.load_model(path)
                    pred = int(model.predict(df.values.reshape(1,-1,1)) > 0.5)
            except Exception as e:
                st.error(f"Model load error {name}: {e}")
                pred = 0

            diagnosis = "No Heart Disease" if pred==0 else "Heart Disease Detected"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{name}</div>
                    <div class="result-text">{diagnosis}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -------------------------
# TAB 2 — BULK PREDICT
# -------------------------
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    up = st.file_uploader("Upload CSV", type="csv")

    if up:
        df = pd.read_csv(up)
        cols = ['Age','Sex','ChestPainType','RestingBP','Cholesterol',
                'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

        if not set(cols).issubset(df.columns):
            st.error("Missing required columns!")
        else:
            maps = {
                "Sex":{"M":0,"F":1,"Male":0,"Female":1},
                "ChestPainType":{"ATA":0,"NAP":1,"ASY":2,"TA":3},
                "RestingECG":{"Normal":0,"ST":1,"LVH":2},
                "ExerciseAngina":{"Y":1,"N":0,"Yes":1,"No":0},
                "ST_Slope":{"Up":0,"Flat":1,"Down":2},
            }
            for col, mp in maps.items():
                if df[col].dtype == object:
                    df[col] = df[col].map(mp)

            model = pickle.load(open("logistic_regression_model.pkl","rb"))
            df["Prediction"] = model.predict(df[cols])

            st.write(df)
            st.markdown(get_downloader_html(df), unsafe_allow_html=True)

# -------------------------
# TAB 3 — MODEL INFO
# -------------------------
with tab3:
    acc = {
        "Logistic Regression":85.86,
        "Decision Tree":80.97,
        "Random Forest":88.04,
        "MLP":75.54,
        "CNN":86.95,
    }
    df_acc = pd.DataFrame({"Model": list(acc.keys()), "Accuracy": list(acc.values())})
    fig = px.bar(df_acc, x="Model", y="Accuracy", text="Accuracy")
    st.plotly_chart(fig)
