from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import en_core_web_sm

# Load the NLP model directly into memory
nlp = en_core_web_sm.load()

app = FastAPI(title="Enterprise NLP Fact Checker")

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
    Uses Named Entity Recognition (NER) to understand the grammatical structure 
    and isolate only verifiable entities (Orgs, Locations, Money, Percentages).
    """
    doc = nlp(text)
    
    # Target specific grammatical entities
    target_labels = {"ORG", "GPE", "PERCENT", "MONEY", "DATE", "CARDINAL", "QUANTITY", "EVENT"}
    entities = [ent.text for ent in doc.ents if ent.label_ in target_labels]
    
    # If the text is sparse, fallback to core noun chunks (e.g., "groundwater depletion")
    if not entities:
        entities = [chunk.text for chunk in doc.noun_chunks]
        
    if not entities:
        return text[:50] # Failsafe
        
    query = " ".join(entities)
    
    # Clean duplicates and limit to 10 words for search API safety
    words = list(dict.fromkeys(query.split()))
    return " ".join(words[:10])

@app.post("/factcheck")
def fact_check(payload: TextPayload):
    raw_text = payload.text
    if len(raw_text.split()) < 4:
         return {"error": "Text segment too brief for NLP processing."}

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
