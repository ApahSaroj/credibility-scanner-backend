from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Authenticity API")

# Allows your Chrome extension to securely talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(payload: TextPayload):
    text = payload.text.lower()
    
    # 1. AI Generation Buzzwords
    ai_buzzwords = ["delve", "testament", "fast-paced", "transformative", "crucial", "tapestry", "demystify", "revolutionize"]
    ai_flags = [word for word in ai_buzzwords if word in text]
    
    # 2. Propaganda / Logical Fallacies / Engagement Bait
    propaganda_phrases = ["you either", "will die if you don't", "secret they don't want you to know", "stop doing this now"]
    propaganda_flags = [phrase for phrase in propaganda_phrases if phrase in text]
    
    # 3. Ghost Citations (Vague data backing without proof)
    ghost_citations = ["studies show", "experts agree", "research proves", "data shows"]
    citation_flags = [phrase for phrase in ghost_citations if phrase in text]
    
    # 4. Legitimate Data Backing Check (Looks for actual numbers/metrics)
    has_numbers = any(char.isdigit() for char in text)
    
    # Mathematical Scoring System
    human_score = 100 - (len(ai_flags) * 15) - (len(propaganda_flags) * 20) - (len(citation_flags) * 15)
    human_score = max(0, min(100, human_score))
    
    # Determine Verdict
    if human_score < 50:
        verdict = "🚨 High Risk: Likely AI-generated or High Fluff"
    elif human_score < 75:
        verdict = "⚠️ Moderate Risk: Contains hype or unverified claims"
    else:
        verdict = "✅ Low Risk: Appears human and authentic"
        
    return {
        "human_score": human_score,
        "data_backed": "Yes (Numbers/Metrics detected)" if has_numbers else "No raw numbers or data points found",
        "flags_detected": ai_flags + propaganda_flags + citation_flags,
        "verdict": verdict
    }
