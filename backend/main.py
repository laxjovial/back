# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import routers
from backend.api.chat_api import router as chat_router
from backend.api.document_api import router as document_router
from backend.api.tool_api import router as tool_router # Assuming you have a tool router

# Import config_manager (assuming it's a singleton or globally accessible instance)
from config.config_manager import config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Assistant Backend",
    description="Backend for the AI Assistant, providing chat, document, and tool functionalities.",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Removed the problematic line ---
# config_manager.load_config() # This line caused the AttributeError

# Include API routers
app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(document_router, prefix="/documents", tags=["Documents"])
app.include_router(tool_router, prefix="/tools", tags=["Tools"]) # Include the tool router

@app.get("/")
async def root():
    return {"message": "AI Assistant Backend is running!"}

# You might want to add an event listener for startup if any global setup is needed
import webbrowser

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    # Any other startup logic, e.g., database connections, initial data loading
    # if config_manager needs explicit initialization, do it here if not already done by import
    # Example: if config_manager had a static init method: ConfigManager.initialize_global_config()
    webbrowser.open_new_tab("http://127.0.0.1:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    # Any cleanup logic, e.g., closing database connections

