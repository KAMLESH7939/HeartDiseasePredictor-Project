import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import tensorflow as tf
import plotly.express as px
# Custom CSS for dark purple theme
st.markdown("""
<style>

    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background-image: url('abstract-gradient-neon-lights.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    /* Remove white layer on top */
    [data-testid="stAppViewBlockContainer"] {
        background: transparent !important;
    }

    /* Make body transparent to show background */
    html, body {
        background-color: transparent !important;
    }

    /* CENTER FORM CONTENT NICELY */
    .block-container {
        max-width: 850px;
        margin: auto;
        padding-top: 40px;
    }

    /* Inputs styling */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        background: rgba(0,0,0,0.45) !important;
        color: #fff !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 0, 255, 0.25);
        padding: 8px;
    }

    label, p, span, div {
        color: #f8d9ff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,0,0,0.55);
        backdrop-filter: blur(8px);
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #dab0ff !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #fff !important;
        border-bottom: 3px solid #ff00ff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.50) !important;
        backdrop-filter: blur(10px);
    }

    /* Heading Colors */
    h1, h2, h3, h4 {
        color: #ffd6ff !important;
    }

    /* ----------------------------
       GLASS CARD RESULT OUTPUT BOX
       ---------------------------- */
    .result-card {
        background: rgba(20, 20, 35, 0.55);
        border-radius: 18px;
        padding: 22px;
        margin-top: 25px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 0, 255, 0.18);
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.18);
        transition: 0.25s ease-in-out;
    }

    .result-card:hover {
        transform: translateY(-6px);
        background: rgba(30, 30, 50, 0.65);
        box-shadow: 0 0 35px rgba(255, 0, 255, 0.40);
    }

    .result-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffb3ff;
        margin-bottom: 10px;
    }

    .result-text {
        color: #ffffff;
        font-size: 18px;
        opacity: 0.9;
    }

</style>
""", unsafe_allow_html=True)


def get_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_heart_disease.csv">Download CSV File</a>'
    return href

st.title("Heart Disease Prediction")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# ============================== TAB 1 ==============================
with tab1:
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Cholesterol (in mg/dl)", min_value=0, max_value=1000)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, format="%.1f")
    st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

    # Categorical conversion
    sex = 0 if sex == "Male" else 1
    chest_pain_dict = {"Typical Angina": 3, "Atypical Angina": 0, "Non-anginal Pain": 1, "Asymptomatic": 2}
    chest_pain = chest_pain_dict[chest_pain]
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1
    resting_ecg_dict = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
    resting_ecg = resting_ecg_dict[resting_ecg]
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    st_slope = st_slope_dict[st_slope]

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    algonames = ["Logistic Regression", "Decision Tree", "Random Forest", "MLP", "CNN"]
    modelnames = ["logistic_regression_model.pkl", "decision_tree_model.pkl",
                  "random_forest_model.pkl", "mlp_model.keras", "cnn_1d_model.keras"]

    predictions = []

    def predict_heart_disease(data):
        for i, modelname in enumerate(modelnames):
            if modelname.endswith('.pkl'):
                model = pickle.load(open(modelname, 'rb'))
                prediction = model.predict(data)
            elif modelname.endswith('.keras'):
                model = tf.keras.models.load_model(modelname)
                prediction = model.predict(data.values.reshape(1, -1, 1) if 'cnn' in modelname else data.values)
                prediction = (prediction > 0.5).astype(int)
            predictions.append(prediction)
        return predictions

    if st.button("Predict"):
        st.subheader("Prediction Results:")
        st.markdown('----------------------------')

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):

          diagnosis = "No Heart Disease Detected" if result[i][0] == 0 else "Heart Disease Detected"

          st.markdown(f"""
           <div class="result-card">
             <div class="result-title">{algonames[i]}</div>
             <div class="result-text">{diagnosis}</div>
           </div>
          """, unsafe_allow_html=True)

# ============================== TAB 2 ==============================
with tab2:
    st.title("Upload CSV for Bulk Prediction")

    st.subheader("Instructions:")
    st.info("""
    Your CSV MUST contain these columns:

    Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

    The app will automatically convert:
    - Sex: M/F → 0/1
    - ChestPainType: TA/ATA/NAP/ASY → 3/0/1/2
    - RestingECG: Normal/ST/LVH → 0/1/2
    - ExerciseAngina: Y/N → 1/0
    - ST_Slope: Up/Flat/Down → 0/1/2
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        model = pickle.load(open("logistic_regression_model.pkl", 'rb'))

        expected_columns = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
            'Oldpeak', 'ST_Slope'
        ]

        # Check for required columns
        if not set(expected_columns).issubset(data.columns):
            st.warning("The uploaded CSV does NOT contain the required columns.")
            st.stop()

        # -----------------------------
        # AUTO-ENCODING STARTS HERE
        # -----------------------------

        # Encoding dictionaries
        sex_map = {"M": 0, "F": 1, "Male": 0, "Female": 1}
        cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
        ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
        angina_map = {"Y": 1, "N": 0, "Yes": 1, "No": 0}
        slope_map = {"Up": 0, "Flat": 1, "Down": 2}

        # Apply mapping only if column contains strings
        if data['Sex'].dtype == object:
            data['Sex'] = data['Sex'].map(sex_map)

        if data['ChestPainType'].dtype == object:
            data['ChestPainType'] = data['ChestPainType'].map(cp_map)

        if data['RestingECG'].dtype == object:
            data['RestingECG'] = data['RestingECG'].map(ecg_map)

        if data['ExerciseAngina'].dtype == object:
            data['ExerciseAngina'] = data['ExerciseAngina'].map(angina_map)

        if data['ST_Slope'].dtype == object:
            data['ST_Slope'] = data['ST_Slope'].map(slope_map)

        # Convert entire dataframe to float
        data = data.astype(float)

        # -----------------------------
        # MAKE PREDICTIONS
        # -----------------------------
        data['Prediction LR'] = ''

        for i in range(len(data)):
            arr = data.iloc[i, :-1].values.astype(float)
            data.loc[i, 'Prediction LR'] = model.predict([arr])[0]

        # Save CSV
        data.to_csv("PredictedHeart.csv", index=False)

        st.subheader("Prediction Results:")
        st.write(data)

        st.markdown(get_downloader_html(data), unsafe_allow_html=True)

    else:
        st.info("Awaiting CSV file upload...")


# ============================== TAB 3 ==============================
with tab3:
    data = {
        'Logistic Regression': 85.86,
        'Decision Tree': 80.97,
        'Random Forest': 88.04,
        'MLP': 75.54,
        'CNN': 86.95
    }
    models = list(data.keys())
    accuracy = list(data.values())
    df = pd.DataFrame(list(zip(models, accuracy)), columns=['Models', 'Accuracy'])
    fig = px.bar(df, x='Models', y='Accuracy', color='Accuracy',
                 text='Accuracy', title='Model Accuracy Comparison')
    st.plotly_chart(fig)
