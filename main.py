from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Live Web Fact-Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

def extract_search_query(text):
    # Cleans the highlighted text to create a search-engine friendly query
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"the", "is", "in", "and", "to", "a", "of", "it", "that", "this", "on", "for", "with", "as", "are", "was", "were"}
    query_words = [w for w in words if w not in stopwords]
    return " ".join(query_words[:10]) # Keep it to 10 words so the search engine doesn't break

@app.post("/factcheck")
def fact_check(payload: TextPayload):
    text = payload.text
    if len(text.split()) < 4:
         return {"error": "Text too short to perform a web verification. Highlight a full sentence."}

    search_query = extract_search_query(text)

    # 1. SCRAPE THE LIVE WEB (Zero API Keys)
    try:
        web_results = []
        with DDGS() as ddgs:
            # Pull the top 3 live web results silently in the background
            results = list(ddgs.text(search_query, max_results=3))
            for r in results:
                web_results.append({"title": r.get('title', ''), "body": r.get('body', ''), "href": r.get('href', '')})
    except Exception as e:
        return {"error": "Live web search was temporarily blocked or timed out. Try again in a moment."}

    if not web_results:
         return {
             "verdict": "Unverified", 
             "confidence_score": "0%", 
             "best_source_title": "None", 
             "best_source_url": "N/A", 
             "best_source_snippet": "No corroborating articles found on the web."
         }

    # 2. MATHEMATICAL SEMANTIC COMPARISON (TF-IDF & Cosine Similarity)
    # This checks how closely the web articles match the claim you highlighted
    corpus = [text] + [res['body'] for res in web_results]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    best_match_index = similarities.argmax()
    max_similarity = similarities[best_match_index]
    best_source = web_results[best_match_index]

    # 3. VERDICT LOGIC
    if max_similarity > 0.15:
        verdict = "✅ Supported by Web Evidence"
    elif max_similarity > 0.05:
        verdict = "⚠️ Partial Match / Needs Context"
    else:
        verdict = "🚨 Unsubstantiated (Contradicts or absent from web)"

    return {
        "verdict": verdict,
        "confidence_score": f"{round(max_similarity * 100, 1)}%",
        "best_source_title": best_source['title'],
        "best_source_url": best_source['href'],
        "best_source_snippet": best_source['body']
    }
