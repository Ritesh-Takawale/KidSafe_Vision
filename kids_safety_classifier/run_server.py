"""
Run Kids Safety Classifier with Waitress WSGI Server
More stable than Flask's built-in development server
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the app
from app import app, load_image_database, TRAINING_DATA_DIR

def main():
    # Create directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs(os.path.join(TRAINING_DATA_DIR, 'safe'), exist_ok=True)
    os.makedirs(os.path.join(TRAINING_DATA_DIR, 'unsafe'), exist_ok=True)
    
    # Load database
    print("[*] Loading image database...")
    db_loaded = load_image_database()
    
    if not db_loaded:
        print("[!] No images found!")
        print("    Add images to:")
        print("    - training_data/safe/")
        print("    - training_data/unsafe/")
    
    # Use waitress for production-ready server
    from waitress import serve
    
    PORT = 8080
    print(f"\n{'='*50}")
    print(f"  KidsSafe AI Server Running!")
    print(f"  Open: http://localhost:{PORT}")
    print(f"{'='*50}\n")
    
    serve(app, host='127.0.0.1', port=PORT, threads=4)

if __name__ == '__main__':
    main()
