from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ai_prediction as ai
import os
import requests

from dotenv import load_dotenv
import os

load_dotenv() 


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
PORT = int(os.getenv("PORT", 10000))



app = FastAPI(title="Career Predictor API + AI Mentor", version="2.0")

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

class MentorInput(BaseModel):
    user_id: str
    message: str


conversation_sessions = {}


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"



@app.get("/")
def health():
    return {"status": "running", "mode": "top3_ensemble + AI mentor"}

@app.post("/predict")
def predict_career(data: StudentInput):
    """Original career predictor endpoint"""
    return ai.predict(data.dict())

@app.post("/ai_mentor")
def ai_mentor(data: MentorInput):
    user_id = data.user_id
    user_message = data.message.strip()

    if not user_message:
        return {"error": "No message provided"}

    if user_id not in conversation_sessions:
        conversation_sessions[user_id] = []

   
    history = conversation_sessions[user_id][-5:]
    formatted_history = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in history])

    
    user_profile = ai.get_user_profile(user_id) if hasattr(ai, "get_user_profile") else None
    profile_text = ""
    if user_profile:
        profile_text = "\n".join([f"{k}: {v}" for k, v in user_profile.items()])
        profile_text = f"User self-assessment scores:\n{profile_text}\n"


    prompt = f"You are an expert career mentor AI.\n{profile_text}Here is the conversation so far:\n{formatted_history}User: {user_message}\nMentor:"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 256
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
       
        reply = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, could not generate response.")
        
    except Exception as e:
        print("AI API failed:", e)
        print("Response text:", getattr(e, 'response', None) and e.response.text)  # Helpful for debugging
        # Fallback response
        reply = "AI mentor is temporarily unavailable. You can still get career recommendations."

  
    conversation_sessions[user_id].append({"role": "user", "content": user_message})
    conversation_sessions[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply, "history": conversation_sessions[user_id]}



PORT = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
