# CardioSurv--Heart-Failure-Survival-Intelligence
CardioSurv predict patient survival outcomes using an ensemble of survival models.
# CardioSurv – Survival Prediction Web App

## 🔗 Live Demo

https://huggingface.co/spaces/Niharika1906/cardiosurv

---

## 📌 Overview

**CardioSurv** is a machine learning-based web application designed to predict patient survival probability using clinical data.
The system is built using survival analysis techniques and trained on a dataset of **299 patients**, enabling it to model time-to-event outcomes and predicts the survial upto 6 months effectively.

It combines statistical and machine learning approaches to provide reliable survival predictions through an easy-to-use web interface.

---

## 🚀 Features

* 📊 Predicts patient survival probability
* ⚡ Fast and interactive web interface
* 🧠 Uses multiple survival analysis models
* 📈 Model performance evaluated using **C-index = 0.8336**
* 🌐 Deployed and accessible online

---

## 🧪 Input Parameters

The model takes clinical inputs such as:

* Age
* Ejection Fraction
* Serum Creatinine
* Blood Pressure
* Other relevant medical features

---

## 📊 Models & Methods Used

### 1. Kaplan–Meier Curve

A non-parametric statistical method used to estimate the survival function from time-to-event data.
It helps visualize survival probability over time and provides an intuitive understanding of patient survival trends.

---

### 2. Cox Proportional Hazards Model

A semi-parametric model used to evaluate the effect of multiple variables on survival time.
It assumes proportional hazards and is widely used in medical survival analysis.

---

### 3. Random Survival Forest (RSF)

An ensemble-based method that captures complex, non-linear relationships in survival data.
It improves prediction robustness compared to traditional models.

---

### 4. Gradient Boosting Survival Analysis (GBSA)

A boosting-based approach that enhances predictive performance by combining multiple weak learners into a strong model.

---

## 📈 Model Performance

The system achieves a **Concordance Index (C-index) of 0.8336**, indicating strong predictive accuracy in ranking survival outcomes.

---

## 👩‍💻 Author

**Niharika Mathankar**

---

## 📜 License

This project is for academic and educational purposes.

