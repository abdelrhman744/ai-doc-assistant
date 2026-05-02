# AI Document Assistant — Full Stack

ChatGPT-style document QA with Arabic + English support.
Stack: FastAPI · Next.js · Qdrant · Ollama (llama3.2 + nomic-embed-text) · Whisper · Tesseract OCR

---

## Prerequisites

| Tool        | Install                                          |
|-------------|--------------------------------------------------|
| Python 3.11+| https://python.org                               |
| Node.js 20+ | https://nodejs.org                               |
| Ollama      | https://ollama.ai                                |
| Qdrant      | https://qdrant.tech/documentation/quick-start/   |
| Tesseract   | `sudo apt install tesseract-ocr tesseract-ocr-ara` (Linux) or https://github.com/UB-Mannheim/tesseract/wiki (Windows) |
| Poppler     | `sudo apt install poppler-utils` (Linux) — needed for pdf2image |
| ffmpeg      | `sudo apt install ffmpeg` — needed for Whisper audio |

---

## 1 · Start Qdrant

```bash
# Docker (easiest)
docker run -d -p 6333:6333 qdrant/qdrant

# Or download binary from https://qdrant.tech
```

## 2 · Pull Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## 3 · Backend

```bash
cd backend

# Create virtual env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend is now live at http://localhost:8000
API docs at http://localhost:8000/docs

## 4 · Frontend

```bash
cd frontend

npm install
npm run dev
```

Frontend is now live at http://localhost:3000

---

## Usage

1. Open http://localhost:3000
2. Upload documents (PDF, DOCX, TXT, images) via the left sidebar
3. Wait for the "chunks added" confirmation
4. Type a question in Arabic or English in the chat box
5. Optionally click **Record voice** to ask by microphone

---

## API Reference

| Method | Endpoint         | Description                              |
|--------|-----------------|------------------------------------------|
| GET    | /api/health      | Health check                             |
| POST   | /api/upload      | Upload files (multipart/form-data)       |
| POST   | /api/chat        | Ask a question (JSON body)               |
| POST   | /api/chat/voice  | Ask by audio (multipart/form-data)       |

### POST /api/chat
```json
// Request
{ "query": "ما هو...", "language": "auto" }

// Response
{ "answer": "...", "sources": "file.pdf (p. 1)", "stt_text": "" }
```

### POST /api/upload
```
Content-Type: multipart/form-data
files: [file1, file2, ...]

Response: { "message": "...", "chunks_added": 42 }
```

---

## Project Structure

```
project/
├── backend/
│   ├── main.py                  # FastAPI app + CORS + routers
│   ├── routes/
│   │   ├── health.py            # GET /api/health
│   │   ├── upload.py            # POST /api/upload
│   │   └── chat.py              # POST /api/chat, /api/chat/voice
│   ├── services/
│   │   ├── db_service.py        # Qdrant client + collection management
│   │   ├── ocr_service.py       # OpenCV + Tesseract OCR (image & PDF)
│   │   ├── audio_service.py     # Whisper STT + ffmpeg preprocessing
│   │   └── rag_service.py       # Full RAG pipeline (query → embed → retrieve → LLM)
│   └── requirements.txt
│
└── frontend/
    ├── app/
    │   ├── layout.tsx
    │   ├── globals.css
    │   └── page.tsx             # Main page (sidebar + chat layout)
    ├── components/
    │   ├── ChatBox.tsx          # Chat history, input, language selector
    │   ├── UploadBox.tsx        # Drag & drop file upload
    │   ├── AnswerBox.tsx        # RTL-aware answer renderer
    │   ├── SourceBox.tsx        # Source file badges
    │   └── VoiceRecorder.tsx    # Mic recording with waveform UI
    ├── services/
    │   └── api.ts               # askQuestion, askVoice, uploadFiles, checkHealth
    ├── next.config.js           # Proxies /api/* → localhost:8000
    ├── tailwind.config.ts
    └── package.json
```

---

## Windows Notes

- Tesseract path is hardcoded in `ocr_service.py`:
  `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`
  Adjust if your install path differs.
- Install Arabic language pack for Tesseract separately.
- ffmpeg must be in your PATH for Whisper audio conversion.
