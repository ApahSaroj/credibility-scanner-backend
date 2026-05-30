from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI(title="Context-Aware Fact Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPayload(BaseModel):
    text: str

def extract_the_gist(text: str) -> str:
    """
    Intelligently isolates the core factual claim by scoring information density,
    ignoring social media fluff and preserving the semantic 'gist'.
    """
    # 1. Strip conversational narrative framing
    fluff_phrases = ["super excited to", "thrilled to share", "proud to announce", "i believe that", "glad to announce", "after months of work"]
    clean_text = text.lower()
    for fluff in fluff_phrases:
        clean_text = clean_text.replace(fluff, "")
        
    # 2. Break paragraph into individual ideas (sentences)
    sentences = re.split(r'[.!?\n]+', text)
    best_sentence = ""
    highest_score = -1
    
    # 3. Score each sentence for Factual Density
    for sentence in sentences:
        if len(sentence.split()) < 3:
            continue
            
        score = 0
        # Massive boost if the sentence contains numbers, percentages, or money (Hard Facts)
        if re.search(r'\d+', sentence):
            score += 15
        # Boost for capitalized words (Proper Nouns, Institutions, Locations)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', sentence))
        score += (proper_nouns * 3)
        
        if score > highest_score:
            highest_score = score
            best_sentence = sentence.strip()
            
    if not best_sentence:
        best_sentence = text 
        
    # 4. Clean the winning 'Gist' sentence for the search engine
    query = re.sub(r'[^\w\s]', '', best_sentence)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "i", "my", "we", "our"}
    words = [w for w in query.split() if w.lower() not in stopwords]
    
    # Return a coherent, continuous phrase (max 10 words for search API limits)
    return " ".join(words[:10])

@app.post("/factcheck")
def fact_check(payload: TextPayload):
    raw_text = payload.text
    if len(raw_text.split()) < 4:
         return {"error": "Text too short. Please highlight a complete sentence."}

    # Extract the true context
    search_query = extract_the_gist(raw_text)

    try:
        web_results = []
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
            for r in results:
                web_results.append({"title": r.get('title', ''), "body": r.get('body', ''), "href": r.get('href', '')})
    except Exception as e:
        return {"error": "Live web search was temporarily blocked. Try again in a moment."}

    if not web_results:
         return {
             "verdict": "🚨 Unsubstantiated", 
             "confidence_score": "0%", 
             "best_source_title": "No Public Verification Found", 
             "best_source_url": "N/A", 
             "best_source_snippet": f"Could not find web evidence for the core claim: '{search_query}'"
         }

    # Semantic comparison between the original text and the web articles
    corpus = [raw_text] + [res['body'] for res in web_results]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_index = similarities.argmax()
    max_similarity = similarities[best_match_index]
    best_source = web_results[best_match_index]

    if max_similarity > 0.12:
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
