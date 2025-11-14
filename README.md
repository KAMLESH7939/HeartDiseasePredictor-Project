# ❤️ Heart Disease Prediction Web App  

### 🧠 Predicting Heart Disease Using Machine Learning & Deep Learning Models  

This project is an **interactive Streamlit web application** that predicts the likelihood of heart disease in a patient based on various medical parameters.  
It supports both **single prediction** (manual input) and **bulk prediction** (CSV upload).  
The app integrates multiple trained models — including traditional ML models and deep learning networks — and provides downloadable prediction results.

---

##  Features  

✅ **Single Patient Prediction:**  
Enter patient details manually to predict heart disease risk using five different models.  

✅ **Bulk Prediction:**  
Upload a CSV file with multiple patient records for batch prediction.  

✅ **Download Results:**  
Export the prediction results as a downloadable CSV file.  

✅ **Model Comparison Dashboard:**  
Visualize model performance using interactive Plotly charts.  

---

##  Models Used  

| Model Name | File Format | Type | Accuracy (%) |
|-------------|-------------|------|---------------|
| Logistic Regression | `.pkl` | Machine Learning | 85.86 |
| Decision Tree | `.pkl` | Machine Learning | 80.97 |
| Random Forest | `.pkl` | Machine Learning | 88.04 |
| MLP (Multi-Layer Perceptron) | `.keras` | Deep Learning | 75.54 |
| CNN (1D Convolutional Neural Network) | `.keras` | Deep Learning | 86.95 |

---

##  Input Features  

| Feature | Description | Type / Encoding |
|----------|--------------|-----------------|
| **Age** | Patient’s age | Integer |
| **Sex** | 0 = Male, 1 = Female | Categorical |
| **ChestPainType** | 3 = Typical Angina, 0 = Atypical Angina, 1 = Non-anginal Pain, 2 = Asymptomatic | Categorical |
| **RestingBP** | Resting blood pressure (mm Hg) | Integer |
| **Cholesterol** | Serum cholesterol (mg/dl) | Integer |
| **FastingBS** | 1 = >120 mg/dl, 0 = ≤120 mg/dl | Categorical |
| **RestingECG** | 0 = Normal, 1 = ST-T abnormality, 2 = Left ventricular hypertrophy | Categorical |
| **MaxHR** | Maximum heart rate achieved | Integer |
| **ExerciseAngina** | 1 = Yes, 0 = No | Categorical |
| **Oldpeak** | ST depression induced by exercise | Float |
| **ST_Slope** | 0 = Upsloping, 1 = Flat, 2 = Downsloping | Categorical |

---

##  How It Works  

1. **Data Preprocessing:**  
   User inputs are converted into numeric form compatible with model training.

2. **Model Loading:**  
   - `.pkl` models are loaded using `pickle` (for scikit-learn models).  
   - `.keras` models are loaded using TensorFlow’s `load_model()`.

3. **Prediction:**  
   - For traditional ML models: `model.predict(data)`  
   - For CNN: data is reshaped to 3D before prediction (`(1, -1, 1)`)  
   - Binary output:  
     - `0` → No Heart Disease  
     - `1` → Heart Disease Detected

4. **Result Display:**  
   Predictions from all models are displayed in a structured format.

5. **Bulk Prediction:**  
   - CSV file is uploaded and processed row-by-row using the Logistic Regression model.  
   - Results are appended as a new column and made available for download.

---

##  Visualization  

The **Model Information** tab provides a visual comparison of all model accuracies using **Plotly Express** bar charts.  
This helps in understanding which model performs best in terms of accuracy.

---

##  Tech Stack  

| Category | Tools / Libraries |
|-----------|-------------------|
| **Frontend UI** | Streamlit |
| **Backend** | Python |
| **Data Handling** | Pandas, NumPy |
| **Model Serialization** | Pickle, TensorFlow |
| **Visualization** | Plotly Express |
| **Encoding / Download** | Base64 |

---

  

