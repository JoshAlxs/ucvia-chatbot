from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 📂 Configurar Jinja2 y carpeta de plantillas
templates = Jinja2Templates(directory="app/templates")

# 📂 Servir archivos estáticos (CSS, JS, imágenes, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 🔹 Ruta de prueba
@app.get("/test")
def test():
    return {"status": "UCVia está funcionando correctamente 🚀"}

# 🔹 Ruta principal: muestra la interfaz
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🔹 Ruta de chat (comunicación con la IA)
@app.post("/chat")
def chat(message: str = Form(...)):
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(message)

    return JSONResponse({"response": response.text})
