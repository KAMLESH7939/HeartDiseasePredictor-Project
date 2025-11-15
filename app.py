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
def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_base64 = get_base64_of_image("gradientt.jpg")

# -------------------------
# Inject CSS (Style 3: Strong Neon Edge)
# -------------------------
st.markdown(
    f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
    color: #f1e9ff;
}}

/* MAIN BACKGROUND */
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

/* CENTER CONTAINER */
.block-container {{
    max-width: 920px;
    margin: auto;
    padding-top: 36px;
    padding-bottom: 80px;
}}

/* ======== INPUT CARD WRAPPER (applies to each field) ======== */
/* Target streamlit's wrapper classes for inputs */
.stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stFileUploader, .stDateInput {{
    /* create the card look */
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 18px;
    position: relative;
    /* two-layer background: inner dark + outer gradient border */
    background-image:
        linear-gradient(rgba(12,10,18,0.75), rgba(12,10,18,0.75)),
        linear-gradient(135deg, rgba(255,0,255,0.22), rgba(120,0,255,0.20) 40%, rgba(255,80,180,0.18));
    background-origin: border-box;
    background-clip: padding-box, border-box;
    border: 2px solid transparent;
    box-shadow:
        0 6px 20px rgba(0,0,0,0.65),
        0 0 26px rgba(255,0,255,0.06) inset;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
}

/* Strong neon edge (outer glowing stroke) via pseudo-element */
.stTextInput:before, .stNumberInput:before, .stSelectbox:before, .stSlider:before, .stFileUploader:before, .stDateInput:before {{
    content: "";
    position: absolute;
    z-index: 0;
    inset: -2px;
    border-radius: 18px;
    background: linear-gradient(90deg, rgba(255,0,255,0.95), rgba(120,0,255,0.95) 40%, rgba(255,80,180,0.95));
    -webkit-mask: linear-gradient(#fff, #fff) content-box, linear-gradient(#fff, #fff);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.45;
    filter: blur(6px);
    transition: opacity 0.28s ease, transform 0.28s ease;
    pointer-events: none;
}}

/* Inner content should appear above the pseudo-element */
.stTextInput > div, .stNumberInput > div, .stSelectbox > div, .stSlider > div, .stFileUploader > div, .stDateInput > div {{
    position: relative;
    z-index: 1;
}}

/* Hover / focus interaction */
.stTextInput:hover, .stNumberInput:hover, .stSelectbox:hover, .stSlider:hover, .stFileUploader:hover, .stDateInput:hover {{
    transform: translateY(-6px);
    box-shadow:
        0 18px 40px rgba(255,0,255,0.10),
        0 6px 30px rgba(0,0,0,0.6);
}}
.stTextInput:hover:before, .stNumberInput:hover:before, .stSelectbox:hover:before, .stSlider:hover:before, .stFileUploader:hover:before, .stDateInput:hover:before {{
    opacity: 0.9;
    transform: scale(1.02);
    filter: blur(8px);
}

/* ======== Label + Text styling inside the card ======== */
/* Make heading/label white and accessible */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label, .stFileUploader label, .stDateInput label {{
    color: #ffffff !important;         /* whitish text as requested */
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
    display: block;
    z-index: 2;
}}

/* The actual input/select boxes: make them transparent and high-contrast */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div,
.stSlider>div>div>input {{
    background: transparent !important;
    color: #ffffff !important;
    border: none !important;
    outline: none !important;
    font-size: 16px !important;
    padding: 10px 12px !important;
    z-index: 2;
}

/* Ensure the select dropdown arrow area still looks good */
.stSelectbox>div>div>div svg {{
    fill: #ffffff !important;
}}

/* Placeholder color */
.stTextInput>div>div>input::placeholder,
.stNumberInput>div>div>input::placeholder {{
    color: rgba(255,255,255,0.6) !important;
}

/* Accessibility: focus state */
.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus,
.stSelectbox>div>div>div:focus {{
    box-shadow: 0 0 12px rgba(255,0,255,0.22);
    outline: none;
    border-radius: 8px;
}

/* ======== Tabs, Buttons, Result cards keep neon style ======== */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(6px);
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 18px;
}}

.stTabs [data-baseweb="tab"] {{
    font-size: 17px !important;
    color: #d8b8ff !important;
    padding: 8px 18px !important;
    transition: 0.2s;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: #fff !important;
    font-weight: 600;
    border-bottom: 3px solid #ff00ff !important;
    text-shadow: 0 0 8px rgba(255,0,255,0.8);
}}

/* Buttons */
.stButton>button {{
    background: linear-gradient(90deg, #ff00ff, #ff44cc) !important;
    color: white !important;
    padding: 10px 24px;
    border-radius: 12px;
    border: none;
    box-shadow: 0 6px 18px rgba(255,0,255,0.16);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    font-weight: 700;
}}

.stButton>button:hover {{
    transform: translateY(-4px) scale(1.03);
    box-shadow: 0 18px 42px rgba(255,0,255,0.24);
}}

/* Result card (kept as before) */
.result-card {{
    background: rgba(22, 18, 40, 0.65);
    border-radius: 20px;
    padding: 24px;
    margin-top: 18px;
    border: 2px solid transparent;
    background-image:
        linear-gradient(rgba(22,18,40,0.80), rgba(22,18,40,0.80)),
        linear-gradient(45deg, #ff00ff, #7700ff, #ff0080);
    background-clip: padding-box, border-box;
    box-shadow: 0 8px 30px rgba(255,0,255,0.08);
}}

.result-title {{
    font-size: 24px;
    color: #ffedf9;
    font-weight: 700;
}}

.result-text {{
    font-size: 18px;
    color: #ffffff;
    margin-top: 8px;
}}

/* Sidebar neon */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.38) !important;
    border-right: 2px solid rgba(255,0,255,0.20);
    backdrop-filter: blur(10px);
}}
[data-testid="stSidebar"] * {{
    color: #ffe6ff !important;
}}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# CSV download helper
# -------------------------
def get_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_heart_disease.csv">Download CSV File</a>'
    return href

# -------------------------
# App UI
# -------------------------
st.title("Cardiac Disease Detection Model")

tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# -------------------------
# TAB 1 - single prediction
# -------------------------
with tab1:
    # Each Streamlit input is visually wrapped by the CSS card above
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    )
    resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = st.number_input("Cholesterol (in mg/dl)", min_value=0, max_value=1000, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, format="%.1f", value=1.0)
    st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

    # Encode categories
    sex_code = 0 if sex == "Male" else 1
    chest_pain_dict = {"Typical Angina": 3, "Atypical Angina": 0, "Non-anginal Pain": 1, "Asymptomatic": 2}
    chest_pain_code = chest_pain_dict[chest_pain]
    fasting_bs_code = 0 if fasting_bs == "<= 120 mg/dl" else 1
    resting_ecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    resting_ecg_code = resting_ecg_dict[resting_ecg]
    exercise_angina_code = 1 if exercise_angina == "Yes" else 0
    st_slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    st_slope_code = st_slope_dict[st_slope]

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_code],
        "ChestPainType": [chest_pain_code],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs_code],
        "RestingECG": [resting_ecg_code],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina_code],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope_code],
    })

    algonames = ["Logistic Regression", "Decision Tree", "Random Forest", "MLP", "CNN"]
    model_files = [
        "logistic_regression_model.pkl",
        "decision_tree_model.pkl",
        "random_forest_model.pkl",
        "mlp_model.keras",
        "cnn_1d_model.keras",
    ]

    def predict_all(data):
        results = []
        for m in model_files:
            if m.endswith(".pkl"):
                model = pickle.load(open(m, "rb"))
                pred = model.predict(data)
            else:
                model = tf.keras.models.load_model(m)
                pred = model.predict(data.values.reshape(1, -1, 1) if "cnn" in m else data.values)
                pred = (pred > 0.5).astype(int)
            results.append(pred)
        return results

    if st.button("Predict"):
        st.subheader("Prediction Results")
        st.markdown("---")
        try:
            preds = predict_all(input_data)
            for i, p in enumerate(preds):
                diagnosis = "No Heart Disease" if p[0] == 0 else "Heart Disease Detected"
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="result-title">{algonames[i]}</div>
                      <div class="result-text">{diagnosis}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Error running models: {{e}}")

# -------------------------
# TAB 2 - bulk predict
# -------------------------
with tab2:
    st.subheader("Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # quick validation
        expected_columns = [
            "Age","Sex","ChestPainType","RestingBP","Cholesterol",
            "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"
        ]
        if not set(expected_columns).issubset(df.columns):
            st.warning("Uploaded CSV is missing required columns.")
        else:
            # load logistic model for bulk predictions
            model = pickle.load(open("logistic_regression_model.pkl", "rb"))

            # mappings
            sex_map = {"M":0,"F":1,"Male":0,"Female":1}
            cp_map = {"ATA":0,"NAP":1,"ASY":2,"TA":3}
            ecg_map = {"Normal":0,"ST":1,"LVH":2}
            angina_map = {"Y":1,"N":0,"Yes":1,"No":0}
            slope_map = {"Up":0,"Flat":1,"Down":2}

            if df["Sex"].dtype == object:
                df["Sex"] = df["Sex"].map(sex_map)
            if df["ChestPainType"].dtype == object:
                df["ChestPainType"] = df["ChestPainType"].map(cp_map)
            if df["RestingECG"].dtype == object:
                df["RestingECG"] = df["RestingECG"].map(ecg_map)
            if df["ExerciseAngina"].dtype == object:
                df["ExerciseAngina"] = df["ExerciseAngina"].map(angina_map)
            if df["ST_Slope"].dtype == object:
                df["ST_Slope"] = df["ST_Slope"].map(slope_map)

            df = df.astype(float)
            df["Prediction"] = df.apply(lambda r: model.predict([r.values])[0], axis=1)

            st.write(df)
            st.markdown(get_downloader_html(df), unsafe_allow_html=True)

# -------------------------
# TAB 3 - model info
# -------------------------
with tab3:
    acc = {
        "Logistic Regression": 85.86,
        "Decision Tree": 80.97,
        "Random Forest": 88.04,
        "MLP": 75.54,
        "CNN": 86.95,
    }
    df_acc = pd.DataFrame({"Model": list(acc.keys()), "Accuracy": list(acc.values())})
    fig = px.bar(df_acc, x="Model", y="Accuracy", color="Accuracy", text="Accuracy", title="Model Accuracy Comparison")
    st.plotly_chart(fig)
