# Naira Fake vs Genuine Note Detector

This project is a Streamlit-based web application for detecting **fake vs genuine Nigerian Naira banknotes** using a deep learning model (MobileNetV2).

## Files
- `naira_fake_detector_google_drive_only.h5` – Trained TensorFlow/Keras model
- `app.py` – Streamlit frontend application
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation

## How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at:
```
http://localhost:8501
```

## Prediction Logic
- **GENUINE**: probability ≥ 0.75
- **FAKE**: probability ≤ 0.25
- **UNCERTAIN**: values in between (manual review recommended)

## Deployment
This app can be deployed on:
- **Streamlit Cloud**
- **Microsoft Azure App Service**

## Disclaimer
This tool assists detection and should not replace official authentication methods.