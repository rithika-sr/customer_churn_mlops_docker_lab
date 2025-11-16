from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import os

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Dockerized FastAPI service with UI for Telco churn prediction.",
    version="1.0.0"
)

# -------------------------------------------------
# Serve UI (Static Files + index.html)
# -------------------------------------------------
app.mount("/ui", StaticFiles(directory="/app/ui"), name="ui")

@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui/index.html", response_class=HTMLResponse)
def serve_ui():
    """
    Serve the user interface for churn prediction.
    """
    html_path = "/app/ui/index.html"
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return f.read()
    return "<h3>UI file not found inside container.</h3>"

# -------------------------------------------------
# Load ML Model + Preprocessor
# -------------------------------------------------
MODEL_PATH = "/app/model/model.pkl"
PREPROCESSOR_PATH = "/app/model/preprocess.pkl"
MODEL_VERSION = "1.0.0"

logger.info("Loading model and preprocessor...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

logger.info("Model and preprocessor loaded successfully.")

# -------------------------------------------------
# Pydantic Schema
# -------------------------------------------------
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# -------------------------------------------------
# Endpoints
# -------------------------------------------------

@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.get("/version")
def version():
    return {
        "model_version": MODEL_VERSION,
        "model_type": "LogisticRegression",
        "dataset": "Telco Customer Churn",
        "api_version": app.version,
    }

@app.post("/predict")
def predict(data: CustomerData):
    """
    Predict churn probability for a customer.
    """
    input_dict = data.dict()
    logger.info(f"Prediction request: tenure={input_dict['tenure']}, "
                f"MonthlyCharges={input_dict['MonthlyCharges']}, Contract={input_dict['Contract']}")

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Transform
    X_transformed = preprocessor.transform(df)

    # Predict
    prob = float(model.predict_proba(X_transformed)[0][1])

    logger.info(f"Prediction output: {prob:.4f}")

    return {"churn_probability": prob}
