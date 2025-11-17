# 📊 Customer Churn Prediction – MLOps Dockerized Project

An end-to-end **Machine Learning + MLOps** project for predicting customer churn using the **Telco Customer Churn** dataset.
This project includes:

✔ A full ML pipeline (preprocessing → training → prediction)
✔ A Dockerized **FastAPI** inference service
✔ A clean and interactive **web UI** for real-time predictions
✔ Versioned & reproducible environment
✔ Production-ready folder structure

---

## 🚀 Project Features

### **🔹 Machine Learning Pipeline**

* Logistic Regression model trained on the Telco Churn dataset
* Automated preprocessing (encoding, scaling, missing value handling)
* Saved and versioned model + preprocessing pipeline
* Predicts churn probability for individual customers

### **🔹 FastAPI Backend**

* `/predict` → Predict churn from API or UI
* `/health` → Check model + API status
* `/version` → Model + API versioning metadata
* Serves a static HTML UI
* Logging enabled for production observability

### **🔹 Web-Based UI (User Interface)**

A simple, modern, responsive UI for customer churn prediction:

<img width="1641" height="1413" alt="image" src="https://github.com/user-attachments/assets/f8493479-742f-4fec-8e16-017b65bde615" />


Features:

* Clean form inputs
* Contract type + Payment type dropdowns
* “Predict Churn” button
* Displays churn probability in bold output section

---

## 📁 Project Architecture


```
                             ┌───────────────────────────┐
                             │      Telco Churn CSV      │
                             │   (data/telco_churn.csv)   │
                             └──────────────┬────────────┘
                                            │
                         (used during train │ container)
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │              TRAIN CONTAINER                   │
                    │         (Dockerfile.train + train.py)          │
                    │                                                │
                    │  • Loads raw dataset                           │
                    │  • Preprocessing pipeline (preprocess.py)      │
                    │  • Trains Logistic Regression model            │
                    │  • Saves:                                      │
                    │       - model/model.pkl                         │
                    │       - model/preprocess.pkl                    │
                    └───────────────┬───────────────────────────────┘
                                    │ (shared volume)
                                    │
                             ┌──────▼──────────────────────┐
                             │     MODEL ARTIFACT STORE     │
                             │           /model/            │
                             │   model.pkl + preprocess.pkl │
                             └──────────────┬───────────────┘
                                            │ (loaded into API)
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │                API CONTAINER                   │
                    │      (Dockerfile.api + FastAPI app.py)         │
                    │                                                │
                    │  Endpoints:                                    │
                    │   • `/predict` → Churn Probability             │
                    │   • `/health`  → API & Model status            │
                    │   • `/version` → Model & API metadata          │
                    │   • `/` serves frontend (index.html)           │
                    │                                                │
                    │  Loads model + preprocessor at startup         │
                    └───────────────┬───────────────────────────────┘
                                    │
                            Serves Frontend UI
                                    │
                             ┌──────▼───────────────────┐
                             │     HTML + JS FRONTEND    │
                             │    (api/index.html file)  │
                             │                           │
                             │   • User enters inputs     │
                             │   • Calls `/predict` API   │
                             │   • Displays probability   │
                             └────────────────────────────┘
```


---

## 🐳 Run with Docker 

### **Step 1 — Build & start the full system**

```bash
docker-compose up --build
```

This will:

* Train the model in the `train` container
* Start the FastAPI service
* Serve the UI automatically

---

## 🌐 Access the App

| Service      | URL                                                            |
| ------------ | -------------------------------------------------------------- |
| Web UI       | [http://localhost:8000](http://localhost:8000)                 |
| Predict API  | [http://localhost:8000/predict](http://localhost:8000/predict) |
| Health Check | [http://localhost:8000/health](http://localhost:8000/health)   |
| Version Info | [http://localhost:8000/version](http://localhost:8000/version) |
| Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs)       |

---

## 🧠 Model Details

* **Model Type:** Logistic Regression
* **Dataset:** Telco Customer Churn
* **Version:** 1.0.0
* **Features:**

  * Tenure
  * Monthly Charges
  * Total Charges
  * Contract Type
  * Payment Method
  * Internet Service
  * Tech Support, Online Security, etc.

---

## 🛠 How to Retrain the Model

### Using Docker:

```bash
docker-compose run train
```

### Or manually:

```bash
cd training
python train.py
```

Updated model files will appear inside `/model`.

---

## 🧪 Example API Request

```json
POST /predict
{
  "tenure": 5,
  "MonthlyCharges": 89.65,
  "TotalCharges": 450.0,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

Response:

```json
{
  "churn_probability": 0.74
}
```

---


## ⭐ Acknowledgments

Dataset sourced from the **IBM Telco Customer Churn dataset**.

