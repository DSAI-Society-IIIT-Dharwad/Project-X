#!/usr/bin/env python3
"""
Launcher script to run both backend and frontend
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def run_backend():
    """Run the FastAPI backend"""
    backend_path = Path(__file__).parent / "backend"
    os.chdir(backend_path)
    
    print("🚀 Starting Backend API Server...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api_hybrid:app", 
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])

def run_frontend():
    """Run the React frontend"""
    frontend_path = Path(__file__).parent / "frontend"
    os.chdir(frontend_path)
    
    print("🎨 Starting Frontend Development Server...")
    return subprocess.Popen(["npm", "run", "dev"])

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🚀 PROJECT X - AI News Assistant")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = run_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend_process = run_frontend()
        
        print("\n" + "=" * 60)
        print("✅ Both servers are running!")
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 60)
        print("\nPress Ctrl+C to stop both servers...")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        
    finally:
        # Cleanup
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
            print("✅ Backend stopped")
            
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
            print("✅ Frontend stopped")
            
        print("👋 All servers stopped!")

if __name__ == "__main__":
    main()