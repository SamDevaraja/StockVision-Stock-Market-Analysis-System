"""
Setup script for StockVision
Initializes the database and creates necessary directories
"""

import os
from backend.database import init_db

def setup_project():
    """Initialize the StockVision project"""
    print("Setting up StockVision...")
    
    # Create necessary directories
    directories = ['uploads', 'models', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created directory: {directory}/")
    
    # Initialize database
    try:
        init_db()
        print("[OK] Database initialized successfully")
    except Exception as e:
        print(f"[ERROR] Error initializing database: {e}")
        return False
    
    print("\n[SUCCESS] StockVision setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the backend API: cd backend && python app.py")
    print("2. In another terminal, start the frontend: streamlit run frontend/app.py")
    print("3. Run tests: python tests/test_api.py")
    
    return True

if __name__ == "__main__":
    setup_project()

