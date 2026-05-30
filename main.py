from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Resilient Social Media Fact-Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

def clean_social_query(text: str, aggressive: bool = False) -> str:
    """
    Advanced text processing to remove conversational social media fluff.
    """
    # Universal conversational phrases often found on LinkedIn/Twitter
    fluff_patterns = [
        r"super excited to announce", r"thrilled to share", r"proud to share",
        r"honored to be", r"after months of hard work", r"delighted to",
        r"i am glad to", r"excited to", r"finally deployed", r"pleased to announce"
    ]
    
    query = text.lower()
    for pattern in fluff_patterns:
        query = re.sub(pattern, "", query)
        
    words = re.findall(r'\b\w+(?:%\+)?\b', query)
    
    stopwords = {
        "the", "is", "in", "and", "to", "a", "of", "it", "that", "this", "on", "for", 
        "with", "as", "are", "was", "were", "our", "my", "i", "we", "you", "your", 
        "about", "finally", "just", "very", "so", "actually", "from", "by", "an"
    }
    
    filtered_words = [w for w in words if w not in stopwords]
    
    # Aggressive Mode: Keep only numbers, metrics, percentages, and long industry terms
    if aggressive:
        filtered_words = [
            w for w in filtered_words 
            if any(char.isdigit() for char in w) or len(w) > 5 or w.endswith('%')
        ]
        
    return " ".join(filtered_words[:8])

def execute_safe_search(query_string: str):
    """
    Executes search extraction wrapper safely with error handling
    """
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query_string, max_results=3))
    except Exception:
        return []

@app.post("/factcheck")
def fact_check(payload: TextPayload):
    raw_text = payload.text
    if len(raw_text.split()) < 4:
         return {"error": "Text segment too brief for meaningful cross-referencing."}

    # STAGE 1: Standard Search
    search_query = clean_social_query(raw_text, aggressive=False)
    results = execute_safe_search(search_query)
    
    # STAGE 2: If Stage 1 fails, trigger Aggressive Query Relaxation
    if not results:
        search_query = clean_social_query(raw_text, aggressive=True)
        results = execute_safe_search(search_query)

    # Convert results safely into standard format
    web_results = []
    if results:
        for r in results:
            web_results.append({
                "title": r.get('title', ''), 
                "body": r.get('body', ''), 
                "href": r.get('href', '')
            })

    # If completely empty after fallbacks
    if not web_results:
         return {
             "verdict": "🚨 Unsubstantiated Narrative", 
             "confidence_score": "0%", 
             "best_source_title": "No Public Verification Document Found", 
             "best_source_url": "N/A", 
             "best_source_snippet": "This claim likely represents private company performance, localized anecdotes, or unindexed claims that are absent from formal public documentation."
         }

    # MATHEMATICAL VECTOR MATCHING
    corpus = [raw_text] + [res['body'] for res in web_results]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_index = similarities.argmax()
    max_similarity = similarities[best_match_index]
    best_source = web_results[best_match_index]

    if max_similarity > 0.12:
        verdict = "✅ Supported by External Web Sources"
    elif max_similarity > 0.04:
        verdict = "⚠️ Context-Dependent / Partial Alignment"
    else:
        verdict = "🚨 Unsubstantiated Claims"

    return {
        "verdict": verdict,
        "confidence_score": f"{round(max_similarity * 100, 1)}%",
        "best_source_title": best_source['title'],
        "best_source_url": best_source['href'],
        "best_source_snippet": best_source['body']
    }
