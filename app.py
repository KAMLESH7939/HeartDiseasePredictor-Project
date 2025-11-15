import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px

# ==========================================================
#            PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Cardiac Disease Detection", layout="wide")

# ==========================================================
#            CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* ------------ CENTERED MAIN CONTAINER (REDUCED WIDTH) ------------ */
.main-container {
    max-width: 1350px;
    margin: auto;
}

/* ------------ BACKGROUND ------------ */
[data-testid="stAppViewContainer"] {
    background-image: url('gradientt.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ------------ TITLE ------------ */
h1 {
    text-align: center !important;
    margin-top: 40px !important;
    margin-bottom: 30px !important;
    font-size: 55px !important;
    font-weight: 700 !important;
    color: #ffdbff !important;
    text-shadow: 0 0 20px rgba(255, 0, 180, 0.7);
}

/* ------------ NEON CAPSULE TABS (NO HOVER EFFECT) ------------ */
.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: center !important;
    gap: 18px !important;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(40, 0, 60, 0.55);
    padding: 10px 28px !important;
    border-radius: 30px !important;
    border: 1px solid rgba(255, 0, 200, 0.35);
    color: #ffd6ff !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    transition: none !important;
}

/* ACTIVE TAB */
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff00cc, #9900ff) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(255,0,200,0.7);
}

/* ------------ INPUT CARD ------------ */
.neon-card {
    width: 100%;
    padding: 22px;
    border-radius: 35px;
    background: linear-gradient(90deg, rgba(80,0,120,0.7), rgba(20,0,40,0.6));
    border: 2px solid rgba(255,0,200,0.5);
    box-shadow: 0 0 25px rgba(255,0,220,0.2);
    margin-top: 28px;
}

.neon-card label {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* ------------ PREDICT BUTTON ------------ */
.predict-btn {
    display: block;
    margin: auto;
    margin-top: 35px;
    background: linear-gradient(90deg, #ff00cc, #9900ff);
    color: white !important;
    padding: 14px 45px;
    font-size: 22px;
    font-weight: 600;
    border-radius: 40px;
    border: none;
    cursor: pointer;
    box-shadow: 0 0 25px rgba(255,0,200,0.5);
}

/* ------------ RESULT CARD ------------ */
.result-card {
    width: 100%;
    margin-top: 28px;
    padding: 22px;
    border-radius: 35px;
    background: linear-gradient(90deg, rgba(80,0,120,0.7), rgba(20,0,40,0.6));
    border: 2px solid rgba(255,0,200,0.6);
    box-shadow: 0 0 40px rgba(255,0,200,0.25);
}

.result-title {
    font-size: 26px;
    color: #ffb3ff;
    font-weight: 700;
}

.result-text {
    font-size: 20px;
    color: #ffffff;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
#         MAIN WRAPPER FOR CONTENT WIDTH
# ==========================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ==========================================================
#                     TITLE
# ==========================================================
st.title("Cardiac Disease Detection Model")

# ==========================================================
#                     TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# ==========================================================
#                     TAB 1
# ==========================================================
with tab1:

    # INPUT CARDS
    def input_card(label, widget):
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        val = widget
        st.markdown('</div>', unsafe_allow_html=True)
        return val

    age = input_card("Age", st.number_input("Age", min_value=1, max_value=120))
    sex = input_card("Sex", st.selectbox("Sex", ["Male", "Female"]))
    chest_pain = input_card("Chest Pain Type",
        st.selectbox("Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]))

    resting_bp = input_card("Resting Blood Pressure",
        st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300))

    cholesterol = input_card("Cholesterol",
        st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000))

    fasting_bs = input_card("Fasting Blood Sugar",
        st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"]))

    resting_ecg = input_card("Resting ECG",
        st.selectbox("Resting ECG", ["Normal",
                                     "ST-T wave abnormality",
                                     "Left ventricular hypertrophy"]))

    max_hr = input_card("Max Heart Rate",
        st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202))

    exercise_angina = input_card("Exercise Angina",
        st.selectbox("Exercise Induced Angina", ["Yes", "No"]))

    oldpeak = input_card("Oldpeak",
        st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, format="%.1f"))

    st_slope = input_card("ST Slope",
        st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"]))

    # ENCODING
    sex = 0 if sex == "Male" else 1
    chest_pain = {"Typical Angina":3, "Atypical Angina":0,
                  "Non-anginal Pain":1, "Asymptomatic":2}[chest_pain]

    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1

    resting_ecg = {"Normal":0,"ST-T wave abnormality":1,
                   "Left ventricular hypertrophy":2}[resting_ecg]

    exercise_angina = 1 if exercise_angina == "Yes" else 0

    st_slope = {"Upsloping":0,"Flat":1,"Downsloping":2}[st_slope]

    input_data = pd.DataFrame({
        'Age':[age],'Sex':[sex],'ChestPainType':[chest_pain],
        'RestingBP':[resting_bp],'Cholesterol':[cholesterol],'FastingBS':[fasting_bs],
        'RestingECG':[resting_ecg],'MaxHR':[max_hr],'ExerciseAngina':[exercise_angina],
        'Oldpeak':[oldpeak],'ST_Slope':[st_slope]
    })

    models = [
        ("Logistic Regression","logistic_regression_model.pkl"),
        ("Decision Tree","decision_tree_model.pkl"),
        ("Random Forest","random_forest_model.pkl"),
        ("MLP","mlp_model.keras"),
        ("CNN","cnn_1d_model.keras")
    ]

    predictions = []

    def run_prediction(data):
        for name, file in models:
            if file.endswith(".pkl"):
                mdl = pickle.load(open(file, "rb"))
                pred = mdl.predict(data)
            else:
                mdl = tf.keras.models.load_model(file)
                pred = mdl.predict(data.values.reshape(1,-1,1))
                pred = (pred > 0.5).astype(int)
            predictions.append(pred)
        return predictions

    # Predict button
    if st.button("Predict", type="primary", key="predict", help="", 
                 use_container_width=False):
        results = run_prediction(input_data)

        for i, (name, _) in enumerate(models):
            status = "No Heart Disease" if results[i][0] == 0 else "Heart Disease Detected"

            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{name}</div>
                    <div class="result-text">{status}</div>
                </div>
            """, unsafe_allow_html=True)

# ==========================================================
#                TAB 2 - BULK
# ==========================================================
with tab2:
    st.header("Upload CSV for Bulk Prediction")

    file = st.file_uploader("Upload CSV File", type="csv")

    if file:
        df = pd.read_csv(file)
        mdl = pickle.load(open("logistic_regression_model.pkl","rb"))

        df["Prediction"] = mdl.predict(df.values)

        st.write(df)

# ==========================================================
#                TAB 3 - MODEL INFO
# ==========================================================
with tab3:

    data = {
        'Logistic Regression': 85.86,
        'Decision Tree': 80.97,
        'Random Forest': 88.04,
        'MLP': 75.54,
        'CNN': 86.95
    }

    df = pd.DataFrame({"Models": data.keys(), "Accuracy": data.values()})

    fig = px.bar(df, x="Models", y="Accuracy", text="Accuracy", color="Accuracy",
                 title="Model Accuracy Comparison")
    st.plotly_chart(fig)

# CLOSE MAIN WRAPPER
st.markdown('</div>', unsafe_allow_html=True)



