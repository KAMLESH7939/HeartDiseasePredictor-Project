import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px

# ==========================================================
#            GLOBAL PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Cardiac Disease Detection", layout="wide")

# ==========================================================
#            CUSTOM CSS STYLING
# ==========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* ---------- BACKGROUND ---------- */
[data-testid="stAppViewContainer"] {
    background-image: url('gradientt.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Center title */
h1 {
    text-align: center !important;
    margin-top: 40px !important;
    margin-bottom: 30px !important;
    font-size: 55px !important;
    font-weight: 700 !important;
    color: #ffdbff !important;
    text-shadow: 0 0 20px rgba(255, 0, 180, 0.7);
}

/* ---------- NEON CAPSULE TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-bottom: 35px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(40, 0, 60, 0.5);
    padding: 10px 25px;
    border-radius: 30px;
    border: 1px solid rgba(255, 0, 200, 0.3);
    color: #ffd6ff !important;
    font-size: 17px;
    font-weight: 600;
    transition: 0.25s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 0, 200, 0.2);
    border-color: rgba(255, 0, 200, 0.6);
    box-shadow: 0 0 15px rgba(255, 0, 200, 0.7);
    transform: translateY(-3px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ff00cc, #9900ff) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(255, 0, 200, 0.8);
}

/* ---------- INPUT CARD WRAPPER ---------- */
.neon-card {
    width: 100%;
    padding: 22px;
    border-radius: 35px;
    background: linear-gradient(90deg, rgba(80,0,120,0.7), rgba(20,0,40,0.6));
    border: 2px solid rgba(255,0,200,0.5);
    box-shadow: 0 0 25px rgba(255,0,220,0.2);
    margin-top: 25px;
    transition: 0.3s ease;
}

.neon-card:hover {
    box-shadow: 0 0 40px rgba(255,0,200,0.5);
    transform: translateY(-4px);
}

/* Make the input boxes inside card visible */
.neon-card label {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* ---------- PREDICT BUTTON ---------- */
.predict-btn {
    display: block;
    margin: auto;
    margin-top: 30px;
    background: linear-gradient(90deg, #ff00cc, #9900ff);
    color: white !important;
    padding: 14px 45px;
    font-size: 22px;
    font-weight: 600;
    border-radius: 40px;
    border: none;
    cursor: pointer;
    box-shadow: 0 0 25px rgba(255,0,200,0.5);
    transition: 0.25s ease-in-out;
}

.predict-btn:hover {
    transform: scale(1.08);
    box-shadow: 0 0 40px rgba(255,0,200,0.9);
}

/* ---------- RESULT CARDS ---------- */
.result-card {
    width: 100%;
    margin-top: 25px;
    padding: 22px;
    border-radius: 35px;
    background: linear-gradient(90deg, rgba(80,0,120,0.7), rgba(20,0,40,0.6));
    border: 2px solid rgba(255,0,200,0.6);
    box-shadow: 0 0 40px rgba(255,0,200,0.25);
    transition: 0.3s;
}

.result-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 55px rgba(255,0,200,0.7);
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
#             DOWNLOAD HELPER
# ==========================================================
def get_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predicted_heart_disease.csv">Download CSV File</a>'

# ==========================================================
#                     MAIN TITLE
# ==========================================================
st.title("Cardiac Disease Detection Model")

# ==========================================================
#                     TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# ==========================================================
#                     TAB 1 - PREDICT
# ==========================================================
with tab1:

    # --- Age ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Sex ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        sex = st.selectbox("Sex", ["Male", "Female"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Chest Pain ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest Pain Type", 
            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Resting BP ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Cholesterol ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=1000)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Fasting Blood Sugar ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Resting ECG ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        resting_ecg = st.selectbox("Resting ECG", 
            ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Max HR ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Exercise Angina ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Oldpeak ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, format="%.1f")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ST Slope ---
    with st.container():
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ENCODING
    sex = 0 if sex == "Male" else 1
    chest_pain_dict = {"Typical Angina":3,"Atypical Angina":0,"Non-anginal Pain":1,"Asymptomatic":2}
    chest_pain = chest_pain_dict[chest_pain]
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1
    resting_ecg_dict = {"Normal":0,"ST-T wave abnormality":1,"Left ventricular hypertrophy":2}
    resting_ecg = resting_ecg_dict[resting_ecg]
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope_dict = {"Upsloping":0,"Flat":1,"Downsloping":2}
    st_slope = st_slope_dict[st_slope]

    input_data = pd.DataFrame({
        'Age':[age],'Sex':[sex],'ChestPainType':[chest_pain],
        'RestingBP':[resting_bp],'Cholesterol':[cholesterol],'FastingBS':[fasting_bs],
        'RestingECG':[resting_ecg],'MaxHR':[max_hr],'ExerciseAngina':[exercise_angina],
        'Oldpeak':[oldpeak],'ST_Slope':[st_slope]
    })

    modelnames = [
        ("Logistic Regression","logistic_regression_model.pkl"),
        ("Decision Tree","decision_tree_model.pkl"),
        ("Random Forest","random_forest_model.pkl"),
        ("MLP","mlp_model.keras"),
        ("CNN","cnn_1d_model.keras")
    ]

    predictions = []

    def predict_heart_disease(data):
        for name, modelname in modelnames:
            if modelname.endswith(".pkl"):
                model = pickle.load(open(modelname,"rb"))
                pred = model.predict(data)
            else:
                model = tf.keras.models.load_model(modelname)
                pred = model.predict(data.values.reshape(1,-1,1))
                pred = (pred > 0.5).astype(int)
            predictions.append(pred)
        return predictions

    # PREDICT BUTTON
    if st.button("Predict", key="predict_btn", use_container_width=False):
        st.markdown('<style>.predict-btn{animation: pulse 1.5s infinite;}</style>', unsafe_allow_html=True)
        st.write("")  

        results = predict_heart_disease(input_data)

        for i, (name, _) in enumerate(modelnames):
            diagnosis = "No Heart Disease" if results[i][0] == 0 else "Heart Disease Detected"
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{name}</div>
                    <div class="result-text">{diagnosis}</div>
                </div>
            """, unsafe_allow_html=True)

# ==========================================================
#                TAB 2 - BULK PREDICT
# ==========================================================
with tab2:
    st.header("Upload CSV for Bulk Prediction")
    st.info("""
    CSV must contain: Age, Sex, ChestPainType, RestingBP, Cholesterol,
    FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    """)

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        model = pickle.load(open("logistic_regression_model.pkl","rb"))
        df['Prediction'] = ""

        for i in range(len(df)):
            arr = df.iloc[i,:-1].values.astype(float)
            df.loc[i,'Prediction'] = model.predict([arr])[0]

        st.write(df)
        st.markdown(get_downloader_html(df), unsafe_allow_html=True)

# ==========================================================
#                TAB 3 - MODEL INFO
# ==========================================================
with tab3:
    data = {
        'Logistic Regression':85.86,
        'Decision Tree':80.97,
        'Random Forest':88.04,
        'MLP':75.54,
        'CNN':86.95
    }
    df = pd.DataFrame({"Models":list(data.keys()), "Accuracy":list(data.values())})

    fig = px.bar(df, x="Models", y="Accuracy", text="Accuracy", color="Accuracy",
                 title="Model Accuracy Comparison")
    st.plotly_chart(fig)



