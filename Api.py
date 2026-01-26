from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)


app = FastAPI(title="Loan Approval Prediction API")


class LoanInput(BaseModel):
    no_of_dependents: int
    education: int
    self_employed: int
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: LoanInput):
    try:
        df = pd.DataFrame([data.dict()])

        
        expected_cols = list(model.feature_names_in_)

        
        mapping = {col.strip(): col for col in expected_cols}

       
        df.rename(columns=mapping, inplace=True)

        
        df = df.reindex(columns=expected_cols, fill_value=0)

        
        y_pred = model.predict(df)
        prediction_value = int(y_pred[0])

        prediction = "Approved" if prediction_value == 1 else "Rejected"

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            confidence = float(max(proba[0]))

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "loan_status": prediction,
                "confidence": confidence
            }
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
