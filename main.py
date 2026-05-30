from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yake
import re

app = FastAPI(title="Enterprise Fact Checker (Lightweight NLP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

def extract_surgical_query(text: str) -> str:
    """
    Uses YAKE (Statistical NLP) to isolate the core factual entities 
    and metrics without requiring heavy machine learning models.
    """
    # 1. Strip the social media fluff first
    fluff_phrases = ["super excited", "thrilled to share", "proud to announce", "i believe that", "glad to announce", "after months of work"]
    clean_text = text.lower()
    for fluff in fluff_phrases:
        clean_text = clean_text.replace(fluff, "")

    # 2. Mathematically extract the top 4 most important concepts
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=4)
    keywords = kw_extractor.extract_keywords(clean_text)
    
    if not keywords:
        words = re.findall(r'\b\w+\b', clean_text)
        return " ".join(words[:8])
        
    # 3. Build the core query
    query_parts = [kw[0] for kw in keywords]
    query = " ".join(query_parts)
    
    # 4. Guarantee that hard numbers are included in the search
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
    if numbers:
        query += " " + " ".join(numbers[:2])
        
    # Clean duplicates and limit to 10 words for search engine safety
    final_words = list(dict.fromkeys(query.split()))
    return " ".join(final_words[:10])

@app.post("/factcheck")
def fact_check(payload: TextPayload):
    raw_text = payload.text
    if len(raw_text.split()) < 4:
         return {"error": "Text segment too brief for processing."}

    search_query = extract_surgical_query(raw_text)

    try:
        web_results = []
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
            for r in results:
                web_results.append({"title": r.get('title', ''), "body": r.get('body', ''), "href": r.get('href', '')})
    except Exception as e:
        return {"error": "Search engine timed out. Please try again."}

    if not web_results:
         return {
             "verdict": "🚨 Unsubstantiated", 
             "confidence_score": "0%", 
             "best_source_title": "No Verification Found", 
             "best_source_url": "N/A", 
             "best_source_snippet": f"No web overlap found for core entities: '{search_query}'"
         }

    # TF-IDF Semantic Verification
    corpus = [raw_text] + [res['body'] for res in web_results]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_index = similarities.argmax()
    max_similarity = similarities[best_match_index]
    best_source = web_results[best_match_index]

    if max_similarity > 0.12:
        verdict = "✅ Verified by Web Sources"
    elif max_similarity > 0.05:
        verdict = "⚠️ Partial Context Match"
    else:
        verdict = "🚨 Unsubstantiated Claim"

    return {
        "verdict": verdict,
        "confidence_score": f"{round(max_similarity * 100, 1)}%",
        "best_source_title": best_source['title'],
        "best_source_url": best_source['href'],
        "best_source_snippet": best_source['body']
    }
