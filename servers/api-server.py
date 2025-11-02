from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import secrets
import sqlite3
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from contextlib import asynccontextmanager

# Security
security = HTTPBearer()

# Models
class CodeRequest(BaseModel):
    prompt: str
    language: str = "python"
    max_length: int = 500
    temperature: float = 0.7

class CodeResponse(BaseModel):
    generated_code: str
    status: str
    model: str

class APIKeyResponse(BaseModel):
    api_key: str
    message: str

# Database setup
def init_db():
    conn = sqlite3.connect("api_keys.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')
    conn.commit()
    conn.close()

# Model manager
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
    
    def load_model(self, model_path: str = "model_weights"):
        """Load the model weights - can be local or from Hugging Face"""
        try:
            print("🚀 Loading Zim Coder model...")
            
            # Check if local weights exist, else download from Hugging Face
            if os.path.exists(model_path):
                print(f"📁 Loading from local: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                print("🌐 Downloading from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/deepseek-coder-1.3b-base",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Save locally for next time
                print("💾 Saving model locally...")
                self.tokenizer.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
            
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def generate_code(self, request: CodeRequest):
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Format prompt
        formatted_prompt = f"# Language: {request.language}\n# Instruction: {request.prompt}\n\n"
        
        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9
            )
        
        # Decode and return
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    model_manager.load_model()
    yield
    # Shutdown: Cleanup if needed
    pass

# Create FastAPI app
app = FastAPI(
    title="Zim Coder API",
    description="Free and Open Source AI Code Generator",
    version="1.0.0",
    lifespan=lifespan
)

# API Key validation
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    conn = sqlite3.connect("api_keys.db")
    cursor = conn.cursor()
    cursor.execute("SELECT key FROM api_keys WHERE key = ? AND is_active = TRUE", 
                   (credentials.credentials,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return credentials.credentials

# Routes
@app.get("/")
async def root():
    return {
        "message": "Zim Coder API v1.0", 
        "status": "running",
        "model_loaded": model_manager.model_loaded
    }

@app.post("/v1/generate", response_model=CodeResponse)
async def generate_code(
    request: CodeRequest, 
    api_key: str = Depends(validate_api_key)
):
    """Generate code - main endpoint"""
    try:
        generated_code = model_manager.generate_code(request)
        
        return CodeResponse(
            generated_code=generated_code,
            status="success",
            model="ZimCoder-1.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/v1/api-keys/create", response_model=APIKeyResponse)
async def create_api_key():
    """Create a new API key"""
    new_key = f"zim_{secrets.token_urlsafe(24)}"
    
    conn = sqlite3.connect("api_keys.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO api_keys (key) VALUES (?)", (new_key,))
    conn.commit()
    conn.close()
    
    return APIKeyResponse(
        api_key=new_key,
        message="Save this API key securely - it won't be shown again!"
    )

@app.get("/v1/api-keys/validate")
async def validate_key(api_key: str = Depends(validate_api_key)):
    """Validate an API key"""
    return {"status": "valid", "message": "API key is active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_manager.model_loaded else "unhealthy",
        "model_loaded": model_manager.model_loaded,
        "service": "Zim Coder API"
    }

if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Start server
    print("🌟 Starting Zim Coder API Server...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔑 Get your API key: POST /v1/api-keys/create")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
)
