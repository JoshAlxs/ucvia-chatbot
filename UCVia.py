# UCVia.py
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

# ------------------ CONFIG ------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Define GOOGLE_API_KEY en .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")  # usa AI Studio

SYSTEM_PROMPT = (
    "Eres UCVia, guía universitario digital. "
    "Responde claro y breve, con pasos prácticos si aplica. "
    "Usa un lenguaje actual y profesional. "
    "Si no sabes, dilo y sugiere dónde verificar."
)


# ------------------ CARGAR DATA ------------------
with open("data/faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

preguntas = [item["pregunta"] for item in faqs]
categorias = [item["categoria"] for item in faqs]

# ------------------ CLASIFICADOR ------------------
vec_path = "models/vectorizer.pkl"
clf_path = "models/classifier.pkl"

if not os.path.exists(vec_path) or not os.path.exists(clf_path):
    print("⚙️ Entrenando clasificador...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preguntas)
    clf = LogisticRegression()
    clf.fit(X, categorias)

    os.makedirs("models", exist_ok=True)
    pickle.dump(vectorizer, open(vec_path, "wb"))
    pickle.dump(clf, open(clf_path, "wb"))
else:
    vectorizer = pickle.load(open(vec_path, "rb"))
    clf = pickle.load(open(clf_path, "rb"))

# ------------------ EMBEDDINGS + FAISS ------------------
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

# ------------------ LOOP PRINCIPAL ------------------
print("✅ UCVia iniciado. Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Tú: ")
    if pregunta.strip().lower() == "salir":
        print("UCVia: ¡Hasta luego!")
        break

    # Clasificación
    cat_pred = clf.predict(vectorizer.transform([pregunta]))[0]

    # Recuperación de respuesta más cercana
    retrieved = retrieve(pregunta, k=1)[0]
    contexto = retrieved["respuesta"]

    # Prompt con contexto
    prompt = f"""{SYSTEM_PROMPT}

Categoría detectada: {cat_pred}
Contexto relevante: {contexto}

Pregunta del estudiante: {pregunta}

Respuesta:
"""

    try:
        respuesta = model.generate_content(prompt)
        print("UCVia:", respuesta.text, "\n")
    except Exception as e:
        print("⚠️ Error:", e)
