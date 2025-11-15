import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px

# Load background image for inline CSS
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

background_base64 = get_base64_image("gradientt.jpg")


# -------------------------
#      CUSTOM CSS
# -------------------------
st.markdown(
    f"""
<style>

/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
}}

/* Background */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

html, body {{
    background-color: transparent !important;
}}

/* Center container */
.block-container {{
    max-width: 900px !important;
    margin: auto;
    padding-top: 40px;
}}

/* ----------------------------
   INPUT CARD WRAPPER (Style 3)
------------------------------*/
.input-card {{
    background: linear-gradient(135deg, rgba(30, 0, 45, 0.65), rgba(80, 0, 100, 0.65));
    padding: 22px 22px 28px 22px;
    border-radius: 20px;

    border: 2px solid transparent;
    background-image:
        linear-gradient(rgba(30,0,45,0.68), rgba(40,0,60,0.68)),
        linear-gradient(45deg, #ff00ff, #8800ff, #ff0080);
    background-origin: border-box;
    background-clip: padding-box, border-box;

    box-shadow: 0 0 18px rgba(255,0,255,0.25);
    margin-bottom: 22px;
    transition: 0.3s ease-in-out;
}}

.input-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 0 25px #ff00ff;
}}

/* Labels inside cards */
.input-card label {{
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}}

/* Input fields */
.input-card .stTextInput>div>div>input,
.input-card .stNumberInput>div>div>input,
.input-card .stSelectbox>div>div>div {{
    background: rgba(255,255,255,0.09) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,0,255,0.3) !important;
    padding: 10px;
    transition: 0.3s ease;
}}

.input-card input:hover,
.input-card select:hover {{
    border: 1px solid #ff00ff !important;
    box-shadow: 0 0 8px #ff00ff !important;
}}

/* ----------------------------
     TABS Glow
------------------------------*/
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
    border-radius: 12px;
}}

.stTabs [data-baseweb="tab"] {{
    color: #e8c3ff !important;
    font-size: 17px;
    padding: 10px 20px !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: white !important;
    border-bottom: 3px solid #ff00ff !important;
    text-shadow: 0 0 6px #ff00ff;
}}

/* ----------------------------
   RESULT CARDS + PULSE ANIMATION
------------------------------*/

@keyframes neonPulse {{
    0% {{
        box-shadow: 0 0 14px rgba(255,0,255,0.25),
                    0 0 28px rgba(120,0,255,0.20);
        transform: scale(1.00);
    }}
    50% {{
        box-shadow: 0 0 22px rgba(255,0,255,0.55),
                    0 0 40px rgba(255,80,180,0.35);
        transform: scale(1.015);
    }}
    100% {{
        box-shadow: 0 0 14px rgba(255,0,255,0.25),
                    0 0 28px rgba(120,0,255,0.20);
        transform: scale(1.00);
    }}
}}

.result-card {{
    border-radius: 20px;
    padding: 22px;
    margin-top: 20px;

    background: linear-gradient(135deg, rgba(30, 0, 45, 0.7), rgba(80, 0, 100, 0.7));
    backdrop-filter: blur(10px);

    border: 2px solid transparent;
    background-image:
        linear-gradient(rgba(30,0,45,0.7), rgba(40,0,60,0.7)),
        linear-gradient(45deg, #ff00ff, #8800ff, #ff0080);

    background-origin: border-box;
    background-clip: padding-box, border-box;

    animation: neonPulse 3s ease-in-out infinite;
}}

.result-title {{
    font-size: 24px;
    font-weight: 600;
    color: #ffb3ff;
}}

.result-text {{
    font-size: 18px;
    color: white;
}}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
#   PAGE TITLE
# -------------------------
st.title("Cardiac Disease Detection Model")

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])


# =====================================================================================
#                               TAB 1 – SINGLE PREDICTION
# =====================================================================================
with tab1:

    # Wrap each input field inside the neon card
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        sex = st.selectbox("Sex", ["Male", "Female"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        chest_pain = st.selectbox("Chest Pain Type",
                                  ["Typical Angina", "Atypical Angina",
                                   "Non-anginal Pain", "Asymptomatic"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 1000)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        resting_ecg = st.selectbox("Resting ECG",
                                   ["Normal", "ST-T wave abnormality",
                                    "Left ventricular hypertrophy"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        max_hr = st.number_input("Max Heart Rate Achieved", 60, 202)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, format="%.1f")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Encode values
    sex = 0 if sex == "Male" else 1
    chest_map = {"Typical Angina": 3, "Atypical Angina": 0, "Non-anginal Pain": 1, "Asymptomatic": 2}
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1
    ecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    angina = 1 if exercise_angina == "Yes" else 0
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_map[chest_pain]],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [ecg_map[resting_ecg]],
        "MaxHR": [max_hr],
        "ExerciseAngina": [angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope_map[st_slope]]
    })

    models = ["logistic_regression_model.pkl",
              "decision_tree_model.pkl",
              "random_forest_model.pkl",
              "mlp_model.keras",
              "cnn_1d_model.keras"]

    names = ["Logistic Regression", "Decision Tree", "Random Forest", "MLP", "CNN"]

    def predict_all(data):
        preds = []
        for m in models:
            if m.endswith(".pkl"):
                model = pickle.load(open(m, "rb"))
                preds.append(model.predict(data)[0])
            else:
                model = tf.keras.models.load_model(m)
                raw = model.predict(data.values.reshape(1, -1, 1))
                preds.append(int(raw > 0.5))
        return preds

    if st.button("Predict"):
        results = predict_all(input_data)

        for name, r in zip(names, results):
            diagnosis = "No Heart Disease" if r == 0 else "Heart Disease Detected"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{name}</div>
                    <div class="result-text">{diagnosis}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =====================================================================================
#                               TAB 2 – BULK PREDICT
# =====================================================================================
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        model = pickle.load(open("logistic_regression_model.pkl", "rb"))

        required = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]

        if not set(required).issubset(df.columns):
            st.error("CSV missing required columns!")
            st.stop()

        # Encode
        map_sex = {"M": 0, "F": 1, "Male": 0, "Female": 1}
        map_cp = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
        map_ecg = {"Normal": 0, "ST": 1, "LVH": 2}
        map_angina = {"Y": 1, "N": 0, "Yes": 1, "No": 0}
        map_slope = {"Up": 0, "Flat": 1, "Down": 2}

        for col, mp in [
            ("Sex", map_sex), ("ChestPainType", map_cp),
            ("RestingECG", map_ecg), ("ExerciseAngina", map_angina),
            ("ST_Slope", map_slope)
        ]:
            if df[col].dtype == object:
                df[col] = df[col].map(mp)

        df["Prediction_LR"] = model.predict(df[required])

        st.write(df)

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="Predicted.csv">Download CSV</a>',
                    unsafe_allow_html=True)


# =====================================================================================
#                               TAB 3 – MODEL INFO
# =====================================================================================
with tab3:
    acc = {
        'Logistic Regression': 85.86,
        'Decision Tree': 80.97,
        'Random Forest': 88.04,
        'MLP': 75.54,
        'CNN': 86.95
    }

    df = pd.DataFrame({"Model": acc.keys(), "Accuracy": acc.values()})
    fig = px.bar(df, x="Model", y="Accuracy", color="Accuracy", title="Model Accuracy Comparison")
    st.plotly_chart(fig)

