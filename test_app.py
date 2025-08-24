#!/usr/bin/env python3
"""
Test script untuk menjalankan aplikasi dengan model yang sudah diperbaiki
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, socketio
    print("Successfully imported app modules")
    
    if __name__ == "__main__":
        print("Starting EMONG application with fixed emotion model...")
        print("Access the application at: http://localhost:5000")
        print("=" * 60)
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
        
except Exception as e:
    print(f"Error starting application: {e}")
    import traceback
    traceback.print_exc()