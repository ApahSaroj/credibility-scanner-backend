import torch
import math
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="SOTA Local NLP Authenticity Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a lightweight, open-source model into memory at startup
# DistilGPT2 is optimized for speed and low RAM footprint (approx. 350MB)
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()  # Set model to evaluation mode

class TextPayload(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(payload: TextPayload):
    text = payload.text
    
    # Tokenize input text and convert to tensor format
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Ensure text is long enough for mathematical significance
    if input_ids.shape[1] < 10:
        return {
            "verdict": "⚠️ Text too short for neural profiling",
            "perplexity": 0,
            "human_score": 0,
            "confidence": "Low"
        }
        
    with torch.no_grad():
        # Pass tokens through the local language model to get logits
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Negative log-likelihood loss
        logits = outputs.logits

    # 1. CALCULATE PERPLEXITY (PPL)
    # PPL = exp(loss)
    perplexity = math.exp(loss.item())
    
    # 2. CALCULATE TOKEN RANK DISTRIBUTION (GLTR Logic)
    # Shift logits and targets to align predictions
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    top_10_count = 0
    top_100_count = 0
    total_tokens = shift_labels.shape[1]
    
    # Evaluate where each actual token ranked in the model's prediction matrix
    for i in range(total_tokens):
        token_logits = shift_logits[0, i, :]
        actual_token_id = shift_labels[0, i].item()
        
        # Sort probabilities to discover the true token rank
        sorted_indices = torch.argsort(token_logits, descending=True)
        rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[0].item()
        
        if rank <= 10:
            top_10_count += 1
        if rank <= 100:
            top_100_count += 1

    top_10_ratio = top_10_count / total_tokens
    top_100_ratio = top_100_count / total_tokens

    # 3. VERDICT DECISION MATRIX
    # SOTA thresholds: AI text generates low perplexity and high top-10 predictability
    if perplexity < 25.0 and top_10_ratio > 0.65:
        verdict = "🚨 High AI Congruence (Highly Predictable Token Paths)"
        human_score = int((perplexity / 25.0) * 30)  # Map down to lower score
    elif perplexity > 55.0 or top_10_ratio < 0.40:
        verdict = "✅ Authentically Human (High Token Chaos)"
        human_score = min(100, int(40 + (perplexity / 3.0)))
    else:
        verdict = "⚠️ Mixed / Highly Edited Profile"
        human_score = int(35 + (perplexity / 2.0))

    return {
        "verdict": verdict,
        "perplexity": round(perplexity, 2),
        "top_10_predictability": f"{round(top_10_ratio * 100, 1)}%",
        "human_score": max(0, min(100, human_score))
    }
