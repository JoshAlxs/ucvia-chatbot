# app/chat_logic.py
import os
import json
import pickle
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- CONFIGURACIÓN ----------------
load_dotenv()  # Carga variables desde .env (local)
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("❌ Falta definir GOOGLE_API_KEY en Render o .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = (
    "Eres UCVia, guía universitario digital. "
    "Responde breve, clara y profesionalmente. "
    "Si no sabes la respuesta, indícalo y sugiere una fuente confiable."
)

# ---------------- CARGA DE DATOS ----------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    faqs = json.load(f)

preguntas = [item["pregunta"] for item in faqs]
categorias = [item["categoria"] for item in faqs]

# ---------------- CLASIFICADOR ----------------
VEC_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vectorizer.pkl")
CLF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")

if not os.path.exists(VEC_PATH) or not os.path.exists(CLF_PATH):
    print("⚙️ Entrenando clasificador (primer uso)...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preguntas)
    clf = LogisticRegression()
    clf.fit(X, categorias)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    pickle.dump(vectorizer, open(VEC_PATH, "wb"))
    pickle.dump(clf, open(CLF_PATH, "wb"))
else:
    vectorizer = pickle.load(open(VEC_PATH, "rb"))
    clf = pickle.load(open(CLF_PATH, "rb"))

# ---------------- EMBEDDINGS + FAISS ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [item["respuesta"] for item in faqs]
embs = embedder.encode(doc_texts, convert_to_numpy=True)
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

def retrieve(query, k=1):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [faqs[i] for i in I[0]]

def responder(pregunta: str) -> str:
    """
    Procesa una pregunta del usuario y genera una respuesta usando Gemini.
    """
    try:
        # Clasificación y recuperación contextual
        cat_pred = clf.predict(vectorizer.transform([pregunta]))[0]
        retrieved = retrieve(pregunta, k=1)[0]
        contexto = retrieved["respuesta"]

        prompt = f"""{SYSTEM_PROMPT}

Categoría detectada: {cat_pred}
Contexto relevante: {contexto}

Pregunta del estudiante: {pregunta}

Respuesta:"""

        respuesta = model.generate_content(prompt)
        return respuesta.text.strip()

    except Exception as e:
        return f"⚠️ Error al generar respuesta: {str(e)}"
