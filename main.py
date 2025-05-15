import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader
import tiktoken

# Configuração
CHUNK_SIZE = 3500  # tokens
UPLOAD_DIR = "static"
CHUNK_DIR = "chunks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")
templates = Jinja2Templates(directory="templates")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def split_text_into_chunks(text, max_tokens=CHUNK_SIZE, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        token_len = len(encoding.encode(word))
        if current_tokens + token_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = token_len
        else:
            current_chunk.append(word)
            current_tokens += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chunks": []})

@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Salvar PDF
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Extrair e dividir
    texto = extract_text_from_pdf(filepath)
    chunks = split_text_into_chunks(texto)

    chunk_files = []
    base = os.path.splitext(filename)[0]
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base}_chunk_{i+1}.txt"
        chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk)
        chunk_files.append(chunk_filename)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chunks": chunk_files,
        "filename": filename
    })
