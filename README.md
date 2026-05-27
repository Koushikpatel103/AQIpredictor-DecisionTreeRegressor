# 🌍 AQI Prediction using Random Forest

A Machine Learning web application that predicts **Air Quality Index (AQI)** using **Random Forest Regression** and **Streamlit Dashboard**.

---

# Project Overview

This project predicts **AQI Value** using pollution indicators and environmental data.

Model Used:

- Random Forest Regressor
- GridSearchCV Hyperparameter Tuning
- Streamlit Deployment
- Plotly Interactive Dashboard

---

# Features

## Machine Learning

- Missing value handling
- Categorical to numerical conversion
- Label Encoding
- Train-test split
- Random Forest Regression
- Hyperparameter tuning
- Feature importance
- Model persistence using Joblib

---

## Streamlit Dashboard

- Professional dark UI
- Sidebar controls
- AQI prediction card
- AQI gauge meter
- Pollution category indicator
- Pollutant bar chart
- Radar chart
- Analytics dashboard
- Interactive Plotly graphs
- Multi-tab interface

---

# Dataset Columns

Input Features:

- Country
- City
- CO AQI Value
- CO AQI Category
- Ozone AQI Value
- Ozone AQI Category
- NO2 AQI Value
- NO2 AQI Category
- PM2.5 AQI Value
- PM2.5 AQI Category

Target:

```text
AQI Value
```

---

# Project Structure

```text
AQI_Prediction_RF/
│
├── data/
│   └── AQI.csv
│
├── models/
│   ├── model.pkl
│   └── label_encoders.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train model:

```bash
python src/train.py
```

Model files generated:

```text
models/model.pkl
models/label_encoders.pkl
```

---

# Run Streamlit App

Launch:

```bash
streamlit run app.py
```

---

# Model

Algorithm:

```python
RandomForestRegressor()
```

Hyperparameter Tuning:

```python
GridSearchCV()
```

---

# Tech Stack

- Python
- Scikit-Learn
- Streamlit
- Plotly
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# Future Improvements

- AQI trend forecasting
- Time-series prediction
- City-wise pollution heatmap
- Weather integration
- Live AQI API support
