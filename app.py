


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ai_prediction as ai



app = FastAPI(title="Career Predictor API", version="2.0")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class StudentInput(BaseModel):
    logic: float
    math: float
    creativity: float
    coding: float
    communication: float
    patience: float
    curiosity: float
    attention: float
    risk_taking: float
    visualization: float


@app.get("/")
def health():
    return {"status": "running", "mode": "top3_ensemble"}


@app.post("/predict")
def predict_career(data: StudentInput):
    return ai.predict(data.dict())

import os

PORT = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)