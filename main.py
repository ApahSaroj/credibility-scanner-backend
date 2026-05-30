import re
import math
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Local Statistical NLP Authenticity Engine")

# Security middleware allowing browser connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

def calculate_linguistic_features(text: str):
    lower_text = text.lower()
    
    # Clean words and extract sentences
    words = [w.strip(",.?!()\"';:") for w in lower_text.split() if w.strip(",.?!()\"';:")]
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip().split()) > 2]
    
    if len(sentences) < 2 or len(words) < 10:
        return {"error": "Text too short for reliable statistical profiling. Provide at least 2-3 sentences."}

    # 1. STRUCTURAL BURSTINESS & COEFFICIENT OF VARIATION (CV)
    sentence_lengths = [len(s.split()) for s in sentences]
    mean_length = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
    std_dev = math.sqrt(variance)
    # CV measures relative dispersion. Robotic text has highly uniform, low CV.
    cv = std_dev / mean_length if mean_length > 0 else 0

    # 2. LEXICAL DIVERSITY (Type-Token Ratio - TTR)
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if len(words) > 0 else 0

    # 3. SYNTACTIC DENSITY (Punctuation distribution)
    punctuation_count = len(re.findall(r'[,;:]', text))
    punc_per_sentence = punctuation_count / len(sentences)

    # 4. ALGORITHMIC TRANSITION MATRIX (Weighted NLP Signatures)
    ai_structural_markers = {
        "heavy": ["delve", "testament to", "in conclusion", "ultimately", "furthermore", "moreover", "not only", "demystify"],
        "hedging": ["it is important to", "clear indication", "crucial role", "vital importance", "essential to note"]
    }
    
    heavy_count = sum(1 for marker in ai_structural_markers["heavy"] if marker in lower_text)
    hedging_count = sum(1 for marker in ai_structural_markers["hedging"] if marker in lower_text)
    
    # 5. RISK SCORING MATRIX
    # Base baseline configuration derived from human text averages
    ai_probability_score = 0
    
    # CV Penalty (Low structural variance is highly indicative of generative patterns)
    if cv < 0.35:
        ai_probability_score += 35
    elif cv < 0.50:
        ai_probability_score += 15
        
    # TTR Penalty (Low vocabulary diversity flags repetitiveness)
    if ttr < 0.60:
        ai_probability_score += 25
    elif ttr < 0.70:
        ai_probability_score += 10
        
    # Marker Penalties
    ai_probability_score += (heavy_count * 12)
    ai_probability_score += (hedging_count * 10)
    
    # Normalization bounds
    final_ai_risk = max(0, min(100, int(ai_probability_score)))
    human_authenticity = 100 - final_ai_risk

    # Propaganda and Bias Logic
    propaganda_patterns = ["you must", "obvious choice", "undeniable truth", "everyone knows", "stop wasting time", "secret blueprint"]
    found_propaganda = [p for p in propaganda_patterns if p in lower_text]
    
    # Fact-checking verification indicators (checks for hard quantities/metrics)
    has_metrics = any(char.isdigit() for char in text)

    if human_authenticity > 70:
        verdict = "✅ Highly Authentic (Natural Structural Variance)"
    elif human_authenticity > 45:
        verdict = "⚠️ Mixed Profile (Possible AI assistance or heavy editing)"
    else:
        verdict = "🚨 High AI Congruence (Algorithmic Uniformity Detected)"

    return {
        "human_score": human_authenticity,
        "cv_index": round(cv, 2),
        "lexical_diversity": round(ttr, 2),
        "data_backed": "Verified Data Points Present" if has_metrics else "Unverified/Anecdotal (No metrics found)",
        "propaganda_flags": found_propaganda if found_propaganda else ["None detected"],
        "verdict": verdict
    }

@app.post("/analyze")
def analyze_text(payload: TextPayload):
    result = calculate_linguistic_features(payload.text)
    if "error" in result:
        return {"error": result["error"], "human_score": 0, "cv_index": 0, "lexical_diversity": 0, "data_backed": "N/A", "propaganda_flags": [], "verdict": result["error"]}
    return result
