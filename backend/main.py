from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.quiz import router as quiz_router

from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.health import router as health_router

app = FastAPI(title="AI Document Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(quiz_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
