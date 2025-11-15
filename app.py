import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px

# --------------------------------------------------------
# LOAD BACKGROUND IMAGE AS BASE64
# --------------------------------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

background_base64 = get_base64_image("gradientt.jpg")


# --------------------------------------------------------
# CUSTOM CSS (COMPLETE, ESCAPED, ERROR-FREE)
# --------------------------------------------------------
st.markdown(
    f"""
<style>

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

.block-container {{
    max-width: 900px;
    margin: auto;
    padding-top: 40px;
}}

/* ============================
   INPUT NEON CARD (STYLE 3)
============================ */
.input-card {{
    background: linear-gradient(135deg, rgba(40,0,60,0.7), rgba(80,0,100,0.7));
    padding: 25px 22px 32px 22px;
    border-radius: 28px;

    border: 2px solid transparent;
    background-image:
        linear-gradient(rgba(40,0,60,0.75), rgba(40,0,60,0.75)),
        linear-gradient(45deg, #ff00ff, #8800ff, #ff0080);
    background-origin: border-box;
    background-clip: padding-box, border-box;

    box-shadow: 0 0 20px rgba(255,0,255,0.35);
    margin-bottom: 26px;
    position: relative;
    transition: 0.3s ease-in-out;
}}

.input-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 0 35px #ff00ff;
}}

.input-card label {{
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 500 !important;
    margin-bottom: 8px !important;
}}

/* Make input box float above card */
.input-card .stNumberInput,
.input-card .stTextInput,
.input-card .stSelectbox {{
    position: relative;
    z-index: 10;
}}

.input-card input,
.input-card select {{
    background: rgba(0,0,0,0.45) !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,0,255,0.35) !important;
    padding: 10px !important;
}}

.input-card input:hover,
.input-card select:hover {{
    border-color: #ff00ff !important;
    box-shadow: 0 0 10px #ff00ff !important;
}}

/* ============================
   TABS
============================ */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0,0,0,0.45);
    border-radius: 12px;
}}

.stTabs [data-baseweb="tab"] {{
    color: #e8c3ff !important;
    font-size: 17px;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: white !important;
    border-bottom: 3px solid #ff00ff !important;
}}

/* ============================
   RESULT CARD + GLOW
============================ */

@keyframes neonPulse {{
    0% {{
        box-shadow: 0 0 14px rgba(255,0,255,0.3),
                    0 0 25px rgba(120,0,255,0.25);
    }}
    50% {{
        box-shadow: 0 0 24px rgba(255,0,255,0.6),
                    0 0 45px rgba(255,0,180,0.35);
    }}
    100% {{
        box-shadow: 0 0 14px rgba(255,0,255,0.3),
                    0 0 25px rgba(120,0,255,0.25);
    }}
}}

.result-card {{
    padding: 25px;
    border-radius: 28px;

    background: linear-gradient(
        135deg,
        rgba(40,0,60,0.7),
        rgba(80,0,100,0.7)
    );

    border: 2px solid transparent;
    background-image:
        linear-gradient(rgba(40,0,60,0.7), rgba(40,0,60,0.7)),
        linear-gradient(45deg, #ff00ff, #8800ff, #ff0080);

    background-origin: border-box;
    background-clip: padding-box, border-box;

    animation: neonPulse 3s infinite ease-in-out;
    margin-bottom: 25px;
}}

.result-title {{
    font-size: 24px;
    color: #ffb3ff;
    font-weight: 600;
}}

.result-text {{
    color: white;
    font-size: 18px;
}}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# TITLE
# ============================================================
st.title("Cardiac Disease Detection Model")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])



# ============================================================
# TAB 1 — SINGLE PREDICTION
# ============================================================
with tab1:

    def card_input(label, widget):
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        value = widget(label)
        st.markdown('</div>', unsafe_allow_html=True)
        return value

    # USE THE CARD WRAPPER FOR ALL INPUTS
    age = card_input("Age", lambda x: st.number_input(x, 1, 120))
    sex = card_input("Sex", lambda x: st.selectbox(x, ["Male", "Female"]))
    chest = card_input("Chest Pain Type", lambda x: st.selectbox(
        x, ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    ))
    resting_bp = card_input("Resting Blood Pressure (mm Hg)", lambda x: st.number_input(x, 0, 300))
    cholesterol = card_input("Cholesterol (mg/dl)", lambda x: st.number_input(x, 0, 1000))
    fasting_bs = card_input("Fasting Blood Sugar", lambda x: st.selectbox(x, ["<= 120 mg/dl", "> 120 mg/dl"]))
    resting_ecg = card_input("Resting ECG", lambda x: st.selectbox(
        x, ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
    ))
    max_hr = card_input("Max Heart Rate Achieved", lambda x: st.number_input(x, 60, 202))
    exercise_angina = card_input("Exercise Induced Angina", lambda x: st.selectbox(x, ["Yes", "No"]))
    oldpeak = card_input("Oldpeak (ST Depression)", lambda x: st.number_input(x, 0.0, 10.0, format="%.1f"))
    st_slope = card_input("ST Slope", lambda x: st.selectbox(x, ["Upsloping", "Flat", "Downsloping"]))

    # ENCODING
    sex = 0 if sex == "Male" else 1
    chest_map = {"Typical Angina": 3, "Atypical Angina": 0, "Non-anginal Pain": 1, "Asymptomatic": 2}
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1
    ecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    angina = 1 if exercise_angina == "Yes" else 0
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    df = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_map[chest]],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [ecg_map[resting_ecg]],
        "MaxHR": [max_hr],
        "ExerciseAngina": [angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope_map[st_slope]],
    })

    # MODEL LIST
    models = [
        ("Logistic Regression", "logistic_regression_model.pkl"),
        ("Decision Tree", "decision_tree_model.pkl"),
        ("Random Forest", "random_forest_model.pkl"),
        ("MLP", "mlp_model.keras"),
        ("CNN", "cnn_1d_model.keras")
    ]

    def predict_all(data):
        results = []
        for name, file in models:
            if file.endswith(".pkl"):
                model = pickle.load(open(file, "rb"))
                pred = int(model.predict(data)[0])
            else:
                model = tf.keras.models.load_model(file)
                raw = model.predict(data.values.reshape(1, -1, 1))
                pred = int(raw > 0.5)
            results.append((name, pred))
        return results

    if st.button("Predict"):
        results = predict_all(df)

        for name, pred in results:
            text = "No Heart Disease" if pred == 0 else "Heart Disease Detected"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{name}</div>
                    <div class="result-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )



# ============================================================
# TAB 2 — BULK PREDICT
# ============================================================
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        required = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]

        if not set(required).issubset(df.columns):
            st.error("CSV missing required columns")
            st.stop()

        # Maps
        maps = {
            "Sex": {"M": 0, "F": 1, "Male": 0, "Female": 1},
            "ChestPainType": {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3},
            "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
            "ExerciseAngina": {"Y": 1, "N": 0, "Yes": 1, "No": 0},
            "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}
        }

        for col, mp in maps.items():
            if df[col].dtype == object:
                df[col] = df[col].map(mp)

        model = pickle.load(open("logistic_regression_model.pkl", "rb"))
        df["Prediction_LR"] = model.predict(df[required])

        st.write(df)

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="Predicted.csv">Download CSV</a>',
            unsafe_allow_html=True
        )



# ============================================================
# TAB 3 — MODEL INFO
# ============================================================
with tab3:

    acc = {
        'Logistic Regression': 85.86,
        'Decision Tree': 80.97,
        'Random Forest': 88.04,
        'MLP': 75.54,
        'CNN': 86.95
    }

    df = pd.DataFrame({
        "Model": list(acc.keys()),
        "Accuracy": list(acc.values())
    })

    fig = px.bar(df, x="Model", y="Accuracy", title="Model Accuracy Comparison")
    st.plotly_chart(fig)


