from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Cargar variables de entorno (.env solo en desarrollo)
load_dotenv()

# Inicializar la app
app = FastAPI(title="UCVia Chatbot", description="Asistente IA con FastAPI y Gemini")

# Configurar carpetas estáticas y templates (si usas interfaz web)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Configurar la API de Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ruta base para verificar estado
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "UCVia Chatbot"})

# Endpoint para chat IA
@app.post("/chat")
async def chat(prompt: str = Form(...)):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return JSONResponse({"respuesta": response.text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Endpoint de prueba rápida sin interfaz (por GET)
@app.get("/test")
async def test():
    return {"status": "UCVia está funcionando correctamente 🚀"}
