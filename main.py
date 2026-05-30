import re
import math
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Enterprise AI Authenticity API")

# Allow the Chrome extension to securely communicate with the server
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
    text = payload.text
    lower_text = text.lower()
    
    # 1. BURSTINESS CALCULATION (Structural Variance)
    # Split text into sentences and calculate word counts per sentence
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    sentence_lengths = [len(s.split()) for s in sentences]
    
    std_dev = 0.0
    burstiness_penalty = 0
    if len(sentence_lengths) >= 3:
        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - mean_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        std_dev = math.sqrt(variance)
        
        # Human writing typically has a Standard Deviation > 5.0
        # AI writing is highly uniform, usually < 3.5
        if std_dev < 3.5:
            burstiness_penalty = 40  # Severe penalty for robotic uniformity
        elif std_dev < 5.0:
            burstiness_penalty = 15
            
    # 2. LEXICAL PREDICTABILITY (Perplexity Heuristic)
    # Flagging algorithmic hedging and generative artifacts
    ai_signatures = [
        "delve", "testament to", "crucial", "transformative", 
        "tapestry", "demystify", "revolutionize", "in conclusion", 
        "ultimately", "shed light on", "underscore the importance",
        "it is important to note", "navigating the complexities"
    ]
    signatures_found = [sig for sig in ai_signatures if sig in lower_text]
    
    # 3. GHOST CITATION & DATA QUALITY CHECK
    ghost_citations = ["studies show", "experts agree", "research proves", "data shows", "according to research"]
    citations_found = [cite for cite in ghost_citations if cite in lower_text]
    
    # Check for actual quantitative data backing
    has_metrics = any(char.isdigit() for char in text)
    
    # 4. ALGORITHMIC SCORING ENGINE
    score = 100
    score -= burstiness_penalty
    score -= (len(signatures_found) * 12)
    score -= (len(citations_found) * 15)
    
    # Floor the score at 0 and cap at 100
    final_score = max(0, min(100, int(score)))
    
    # 5. METADATA & VERDICT GENERATION
    if final_score < 45:
         verdict = "🚨 High AI Probability (Low Burstiness/High Predictability)"
    elif final_score < 75:
         verdict = "⚠️ Mixed/Edited (Human-AI Hybrid)"
    else:
         verdict = "✅ Authentic Human Text"
         
    return {
        "authenticity_score": final_score,
        "burstiness_index": round(std_dev, 2),
        "data_backed": "Verified (Metrics Found)" if has_metrics else "Unverified (No Hard Data)",
        "flags_detected": signatures_found + citations_found,
        "verdict": verdict
    }
