#!/usr/bin/env python3
"""
AI News Assistant - Setup Script
Automates the installation and configuration process
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "python": "python3 --version",
        "node": "node --version",
        "npm": "npm --version",
        "pip": "pip --version"
    }
    
    missing = []
    for tool, command in requirements.items():
        if not run_command(command, f"Checking {tool}"):
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing required tools: {', '.join(missing)}")
        print("Please install the missing tools and run this script again.")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def setup_environment():
    """Setup environment file"""
    print("🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Environment file created from template!")
        print("⚠️  Please edit .env file with your API keys before running the system.")
    elif env_file.exists():
        print("✅ Environment file already exists!")
    else:
        print("❌ Environment template not found!")
        return False
    
    return True

def install_backend_dependencies():
    """Install Python dependencies"""
    print("📦 Installing backend dependencies...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def install_frontend_dependencies():
    """Install Node.js dependencies"""
    print("📦 Installing frontend dependencies...")
    
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # Change to frontend directory
    original_cwd = os.getcwd()
    os.chdir(frontend_path)
    
    try:
        success = run_command("npm install", "Installing Node.js packages")
        return success
    finally:
        os.chdir(original_cwd)

def initialize_database():
    """Initialize the database"""
    print("🗄️  Initializing database...")
    
    try:
        # Add project root to path
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root))
        
        # Import and run database initialization
        from backend.db import init_db
        
        async def init():
            await init_db()
        
        asyncio.run(init())
        print("✅ Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def validate_configuration():
    """Validate configuration"""
    print("🔍 Validating configuration...")
    
    try:
        from config import validate_config
        return validate_config()
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating necessary directories...")
    
    directories = [
        "models",
        "models/faiss_index",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully!")
    return True

def main():
    """Main setup function"""
    print("🚀 AI News Assistant - Setup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the project root directory!")
        return False
    
    steps = [
        ("Checking requirements", check_requirements),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Installing backend dependencies", install_backend_dependencies),
        ("Installing frontend dependencies", install_frontend_dependencies),
        ("Initializing database", initialize_database),
        ("Validating configuration", validate_configuration),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_function():
            failed_steps.append(step_name)
    
    print("\n" + "=" * 50)
    
    if failed_steps:
        print("❌ Setup completed with errors!")
        print(f"Failed steps: {', '.join(failed_steps)}")
        print("\nPlease fix the issues and run the script again.")
        return False
    else:
        print("✅ Setup completed successfully!")
        print("\n🎉 Your AI News Assistant is ready!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python launcher.py")
        print("3. Open http://localhost:3000 in your browser")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
