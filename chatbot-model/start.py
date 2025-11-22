"""
Startup script for Mining Laws Chatbot API Server
Automatically loads .env file and starts the server
"""

import os
import sys
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / '.env'
    
    if not env_file.exists():
        print("❌ Error: .env file not found!")
        print("\nPlease create a .env file with:")
        print("export GROQ_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Read and parse .env file
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle export statements
                if line.startswith('export '):
                    line = line[7:]  # Remove 'export '
                
                # Parse KEY=VALUE or KEY='VALUE'
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    
                    # Set environment variable
                    os.environ[key] = value


def check_requirements():
    """Check if all requirements are met"""
    print("="*70)
    print("Mining Laws Chatbot API - Startup Checks")
    print("="*70)
    print()
    
    # Check 1: Load .env file
    print("[1/4] Loading environment variables from .env...")
    load_env_file()
    
    # Check 2: Verify API key
    print("[2/4] Checking GROQ_API_KEY...")
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Error: GROQ_API_KEY not set in .env file!")
        print("\nAdd this line to your .env file:")
        print("export GROQ_API_KEY='your-api-key-here'")
        sys.exit(1)
    print(f"✓ API Key loaded: {api_key[:20]}...")
    
    # Check 3: Verify FAISS index
    print("[3/4] Checking for FAISS index...")
    base_index = Path("models/mining_laws_base.index")
    finetuned_index = Path("models/mining_laws_finetuned.index")
    
    if not base_index.exists() and not finetuned_index.exists():
        print("⚠ Warning: FAISS index not found!")
        print("\nPlease run one of:")
        print("  python src/build_base_index.py")
        print("  python src/finetune_model.py")
        print("\nContinuing anyway (server will show error)...")
    elif finetuned_index.exists():
        print("✓ Fine-tuned FAISS index found")
    else:
        print("✓ Base FAISS index found")
    
    # Check 4: Import check
    print("[4/4] Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI and Uvicorn installed")
    except ImportError as e:
        print(f"❌ Error: Missing dependency - {e}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
    
    print()
    print("="*70)
    print("✓ All checks passed! Starting server...")
    print("="*70)
    print()


def main():
    """Main startup function"""
    # Run checks
    check_requirements()
    
    # Import and run the server
    print("📚 API Documentation:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc:      http://localhost:8000/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    print("="*70)
    print()
    
    # Import and run uvicorn
    import uvicorn
    
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)
