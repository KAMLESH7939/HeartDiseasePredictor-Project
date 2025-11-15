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
# Inject CSS (escaped braces for f-string)
# -------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
    color: #f6ecff;
}}

/* Background */
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

.block-container {{
    max-width: 980px !important;
    margin: auto;
    padding-top: 32px;
    padding-bottom: 80px;
}}

/* -----------------------
   Neon Input Card (Option A)
   Wraps label + widget inside neon card
   ---------------------- */
.input-card {{
    position: relative;
    border-radius: 24px;
    padding: 20px;
    margin-bottom: 20px;

    /* double-layer effect: inner dark + gradient border */
    background-image:
        linear-gradient(rgba(18,12,25,0.85), rgba(18,12,25,0.85)),
        linear-gradient(90deg, rgba(255,0,255,0.95), rgba(120,0,255,0.95) 45%, rgba(255,80,180,0.95));
    background-origin: border-box;
    background-clip: padding-box, border-box;
    border: 2px solid transparent;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6), 0 0 18px rgba(255,0,255,0.12);
    transition: transform 0.28s ease, box-shadow 0.28s ease;
}}

.input-card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 20px 50px rgba(255,0,255,0.16), 0 0 45px rgba(120,0,255,0.10);
}}

/* Make sure label (the text above the widget) is white and visible */
.input-card label {{
    display: block;
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}}

/* Style the actual input/select/number inside the card */
.input-card .stNumberInput>div>div>input,
.input-card .stTextInput>div>div>input,
.input-card .stSelectbox>div>div>div {{
    background: rgba(0,0,0,0.48) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,0,255,0.22) !important;
    padding: 10px !important;
    border-radius: 10px !important;
    outline: none !important;
    font-size: 16px !important;
    transition: box-shadow 0.22s ease, border-color 0.22s ease, transform 0.22s ease;
}}

/* Hover for the inner control */
.input-card .stNumberInput>div>div>input:hover,
.input-card .stTextInput>div>div>input:hover,
.input-card .stSelectbox>div>div>div:hover {{
    border-color: #ff00ff !important;
    box-shadow: 0 0 12px rgba(255,0,255,0.18);
    transform: translateY(-1px);
}}

/* Select arrow color */
.input-card .stSelectbox>div>div>div svg {{
    fill: #ffffff !important;
}}

/* Placeholder color */
.input-card input::placeholder {{
    color: rgba(255,255,255,0.6) !important;
}}

/* Accessibility focus */
.input-card .stNumberInput>div>div>input:focus,
.input-card .stTextInput>div>div>input:focus,
.input-card .stSelectbox>div>div>div:focus {{
    box-shadow: 0 0 14px rgba(255,0,255,0.22);
    border-color: #ff00ff !important;
}}

/* Tabs theme */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(8px);
    border-radius: 10px;
    padding: 6px;
}}

.stTabs [data-baseweb="tab"] {{
    color: #dfb3ff !important;
    font-weight: 500;
    padding: 8px 16px !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: #fff !important;
    border-bottom: 3px solid #ff00ff !important;
    text-shadow: 0 0 6px rgba(255,0,255,0.9);
}}

/* -----------------------
   Predict Button - Neon Stunning
   ---------------------- */
.stButton>button {{
    background: linear-gradient(90deg, #ff00ff 0%, #ff44cc 50%, #ff7ab3 100%) !important;
    color: white !important;
    padding: 12px 30px;
    font-size: 18px;
    border-radius: 14px;
    border: none;
    box-shadow: 0 8px 28px rgba(255,0,255,0.22), inset 0 -2px 8px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    font-weight: 700;
}}

/* Neon button hover + slight pulse */
.stButton>button:hover {{
    transform: translateY(-4px) scale(1.03);
    box-shadow: 0 18px 48px rgba(255,0,255,0.34), 0 0 60px rgba(255,68,204,0.18);
}}

/* button focus (keyboard) */
.stButton>button:focus {{
    outline: none;
    box-shadow: 0 0 16px rgba(255,0,255,0.32);
}}

/* small pulse animation on hover start */
.stButton>button:active {{
    transform: translateY(-2px) scale(1.02);
}}

/* -----------------------
   Result cards + hover + pulse
   ---------------------- */
@keyframes neonPulse {{
    0% {{ box-shadow: 0 0 14px rgba(255,0,255,0.28), 0 0 28px rgba(120,0,255,0.18); transform: scale(1.00); }}
    50% {{ box-shadow: 0 0 30px rgba(255,0,255,0.55), 0 0 50px rgba(255,80,180,0.30); transform: scale(1.01); }}
    100% {{ box-shadow: 0 0 14px rgba(255,0,255,0.28), 0 0 28px rgba(120,0,255,0.18); transform: scale(1.00); }}
}}

.result-card {{
    border-radius: 22px;
    padding: 20px;
    margin-top: 18px;
    background-image:
        linear-gradient(rgba(20,18,35,0.75), rgba(20,18,35,0.75)),
        linear-gradient(45deg, #ff00ff, #7700ff, #ff0080);
    background-origin: border-box;
    background-clip: padding-box, border-box;
    animation: neonPulse 3s ease-in-out infinite;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}

.result-card:hover {{
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0 28px 60px rgba(255,0,255,0.28), 0 0 80px rgba(120,0,255,0.22);
    border-color: #ff00ff;
}}

/* Titles inside results */
.result-title {{
    font-size: 22px;
    color: #ffedff;
    font-weight: 700;
}}

.result-text {{
    font-size: 18px;
    color: #ffffff;
    margin-top: 8px;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.36) !important;
    border-right: 2px solid rgba(255,0,255,0.12);
    backdrop-filter: blur(8px);
}}
[data-testid="stSidebar"] * {{
    color: #ffe6ff !important;
}}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helper: download CSV
# -------------------------
def get_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predicted.csv">Download CSV</a>'

# -------------------------
# App Title
# -------------------------
st.title("Cardiac Disease Detection Model")

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])


# -------------------------
# Helper to render input inside card
# -------------------------
def card_input_number(label, key=None, min_value=None, max_value=None, value=None, step=None, format=None):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    # show label and input inside same card
    if format:
        val = st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, format=format, key=key)
    else:
        val = st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)
    st.markdown('</div>', unsafe_allow_html=True)
    return val

def card_input_select(label, options, key=None):
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    val = st.selectbox(label, options, key=key)
    st.markdown('</div>', unsafe_allow_html=True)
    return val


# -------------------------
# TAB 1: Single Prediction
# -------------------------
with tab1:
    age = card_input_number("Age", min_value=1, max_value=120, value=45)
    sex = card_input_select("Sex", ["Male", "Female"])
    chest_pain = card_input_select("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    resting_bp = card_input_number("Resting Blood Pressure (in mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = card_input_number("Cholesterol (in mg/dl)", min_value=0, max_value=1000, value=200)
    fasting_bs = card_input_select("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = card_input_select("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    max_hr = card_input_number("Max Heart Rate Achieved", min_value=60, max_value=202, value=150)
    exercise_angina = card_input_select("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = card_input_number("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    st_slope = card_input_select("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

    # Encode
    sex_code = 0 if sex == "Male" else 1
    chest_map = {"Typical Angina": 3, "Atypical Angina": 0, "Non-anginal Pain": 1, "Asymptomatic": 2}
    fasting_code = 0 if fasting_bs == "<= 120 mg/dl" else 1
    ecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    exang_code = 1 if exercise_angina == "Yes" else 0
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_code],
        "ChestPainType": [chest_map[chest_pain]],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_code],
        "RestingECG": [ecg_map[resting_ecg]],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exang_code],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope_map[st_slope]],
    })

    model_list = [
        ("Logistic Regression", "logistic_regression_model.pkl"),
        ("Decision Tree", "decision_tree_model.pkl"),
        ("Random Forest", "random_forest_model.pkl"),
        ("MLP", "mlp_model.keras"),
        ("CNN", "cnn_1d_model.keras"),
    ]

    # Predict button in center
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict"):
            # run predictions and show neon cards
            results = []
            for name, fname in model_list:
                try:
                    if fname.endswith(".pkl"):
                        model = pickle.load(open(fname, "rb"))
                        pred = int(model.predict(input_data)[0])
                    else:
                        model = tf.keras.models.load_model(fname)
                        raw = model.predict(input_data.values.reshape(1, -1, 1))
                        pred = int(raw > 0.5)
                except Exception as e:
                    st.error(f"Error loading/running model {name}: {e}")
                    pred = 0
                results.append((name, pred))

            # Render results as neon cards
            for name, pred in results:
                diagnosis = "No Heart Disease" if pred == 0 else "Heart Disease Detected"
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
# TAB 2: Bulk prediction
# -------------------------
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        required_cols = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
        if not set(required_cols).issubset(df.columns):
            st.error("CSV missing required columns.")
        else:
            # simple encoding maps
            maps = {
                "Sex": {"M":0,"F":1,"Male":0,"Female":1},
                "ChestPainType": {"ATA":0,"NAP":1,"ASY":2,"TA":3},
                "RestingECG": {"Normal":0,"ST":1,"LVH":2},
                "ExerciseAngina": {"Y":1,"N":0,"Yes":1,"No":0},
                "ST_Slope": {"Up":0,"Flat":1,"Down":2}
            }
            for col, mp in maps.items():
                if df[col].dtype == object:
                    df[col] = df[col].map(mp)

            model = pickle.load(open("logistic_regression_model.pkl","rb"))
            df["Prediction"] = model.predict(df[required_cols])
            st.write(df)
            st.markdown(get_downloader_html(df), unsafe_allow_html=True)

# -------------------------
# TAB 3: Model info
# -------------------------
with tab3:
    acc = {
        'Logistic Regression': 85.86,
        'Decision Tree': 80.97,
        'Random Forest': 88.04,
        'MLP': 75.54,
        'CNN': 86.95
    }
    df_acc = pd.DataFrame({"Model": list(acc.keys()), "Accuracy": list(acc.values())})
    fig = px.bar(df_acc, x="Model", y="Accuracy", color="Accuracy", text="Accuracy", title="Model Accuracy Comparison")
    st.plotly_chart(fig)


