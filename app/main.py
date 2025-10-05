from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.chat_logic import responder

app = FastAPI(title="UCVia ChatBot")

# Rutas estáticas y plantillas
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal con la interfaz del chat"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_api(request: Request):
    """Recibe mensajes del usuario y devuelve respuesta IA"""
    data = await request.json()
    pregunta = data.get("message", "")
    respuesta = responder(pregunta)
    return JSONResponse({"response": respuesta})
