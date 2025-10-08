from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os
from dotenv import load_dotenv

# 🔹 Cargar variables de entorno (.env en local; Render usa su Environment)
load_dotenv()

app = FastAPI()

# 📂 Configurar Jinja2 y carpeta de plantillas
templates = Jinja2Templates(directory="app/templates")

# 📂 Servir archivos estáticos (CSS, JS, imágenes, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 🔹 Inicializar cliente OpenAI (usa la variable del Dashboard de Render)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔹 Ruta de prueba
@app.get("/test")
def test():
    return {"status": "UCVia está funcionando correctamente 🚀"}

# 🔹 Ruta principal: muestra la interfaz
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🔹 Ruta del chat (comunicación con la IA)
@app.post("/chat")
def chat(message: str = Form(...)):
    try:
        # Usa modelo liviano y rápido (puedes cambiar a gpt-4o)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            max_tokens=512
        )

        respuesta = completion.choices[0].message.content.strip()
        return JSONResponse({"response": respuesta})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
