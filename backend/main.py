"""
TreeSat Analytics Dashboard - FastAPI Backend
Main application entry point.
"""

# uvicorn backend.main:app --reload --port 8000

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes.api import router as api_router
from .model_loader import ModelSingleton

# Create FastAPI app
app = FastAPI(
    title="TreeSat Analytics API",
    description="ML inference service for TreeSat Benchmark Sentinel Dataset analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount preview directory for serving images
previews_dir = os.path.join(os.path.dirname(__file__), "previews")
os.makedirs(previews_dir, exist_ok=True)
app.mount("/previews", StaticFiles(directory=previews_dir), name="previews")

# Include API routes
app.include_router(api_router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup for fast inference."""
    print("[Startup] Pre-loading TreeSat model...")
    try:
        ModelSingleton.get_model()
        print("[Startup] Model loaded successfully!")
    except Exception as e:
        print(f"[Startup] Warning: Model loading failed: {e}")
        print("[Startup] Server will still start, model will be loaded on first request.")


@app.get("/")
async def root():
    return {
        "service": "TreeSat Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /api/upload",
            "predict": "POST /api/predict",
            "preview": "POST /api/preview",
            "batch_stats": "GET /api/batch-stats",
            "health": "GET /api/health"
        }
    }
