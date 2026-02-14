from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ai_prediction as ai
import os
import requests

from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file

# Now you can safely get variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
PORT = int(os.getenv("PORT", 10000))


# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI(title="Career Predictor API + AI Mentor", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Data Models
# -----------------------------
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

# -----------------------------
# In-Memory Session Storage
# -----------------------------
conversation_sessions = {}

# -----------------------------
# Gemini / LLM API Config
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Put your API key in env variable
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = "https://api.gemini.ai/v1/generate"  # Update if Google endpoint changes

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "running", "mode": "top3_ensemble + AI mentor"}

@app.post("/predict")
def predict_career(data: StudentInput):
    """Original career predictor endpoint"""
    return ai.predict(data.dict())

@app.post("/ai_mentor")
def ai_mentor(data: MentorInput):
    """AI mentor endpoint with multi-turn conversation and personalization"""
    user_id = data.user_id
    user_message = data.message.strip()

    if not user_message:
        return {"error": "No message provided"}

    # Initialize session history if first message
    if user_id not in conversation_sessions:
        conversation_sessions[user_id] = []

    # Fetch last 5 messages for context
    history = conversation_sessions[user_id][-5:]  # limit context to last 5 turns

    # Format conversation history
    formatted_history = ""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        formatted_history += f"{role.capitalize()}: {content}\n"

    # Get user profile from ai_prediction (if implemented)
    user_profile = ai.get_user_profile(user_id) if hasattr(ai, "get_user_profile") else None
    profile_text = ""
    if user_profile:
        profile_text = "\n".join([f"{k}: {v}" for k, v in user_profile.items()])
        profile_text = f"User self-assessment scores:\n{profile_text}\n"

    # Build prompt for AI
    prompt = (
        f"You are an expert career mentor AI.\n"
        f"{profile_text}"
        f"Here is the conversation so far:\n"
        f"{formatted_history}"
        f"User: {user_message}\n"
        f"Mentor:"
    )

    # Call Gemini API
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GEMINI_MODEL,
        "prompt": prompt,
        "max_output_tokens": 256
    }

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Adjust parsing based on actual Gemini API response structure
        reply = result.get("response", {}).get("text", "Sorry, I could not generate a response.")
    except Exception as e:
        return {"error": "Failed to connect to AI API", "details": str(e)}

    # Update session history
    conversation_sessions[user_id].append({"role": "user", "content": user_message})
    conversation_sessions[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply, "history": conversation_sessions[user_id]}

# -----------------------------
# Run App
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
