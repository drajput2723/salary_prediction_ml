# 💼 Salary Prediction System

A Machine Learning based web application that predicts employee salary using Age, Gender, Education, Job Title, and Years of Experience.

---

## 🚀 Live Demo

https://salary-prediction-ml-0srb.onrender.com

---

## 📂 GitHub Repository

https://github.com/drajput2723/salary_prediction_ml

---

## ✨ Features

- Predict Employee Salary
- Responsive User Interface
- Monthly & Yearly Salary Calculation
- Salary Level Detection
- Model Performance Table
- Interactive Salary Comparison Chart
- Flask Web Application
- Machine Learning Based Prediction

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML5
- CSS3
- JavaScript
- Chart.js
- Joblib

---

## 🤖 Machine Learning Model

- Algorithm: Random Forest Regressor
- R² Score: 80.69%
- MAE: 16,052
- RMSE: 22,737

---

## 📁 Project Structure

```text
salary_prediction_ml/
│
├── app.py
├── main.py
├── salary_model.pkl
├── feature_columns.pkl
├── gender_encoder.pkl
├── education_encoder.pkl
├── job_encoder.pkl
├── job_titles.pkl
├── requirements.txt
├── Procfile
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/drajput2723/salary_prediction_ml.git

cd salary_prediction_ml

pip install -r requirements.txt

python app.py
```