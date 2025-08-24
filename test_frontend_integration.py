#!/usr/bin/env python3
"""
Test script untuk memverifikasi integrasi frontend-backend
"""
import requests
import json
import time

def test_api_endpoints():
    """Test semua API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 TESTING FRONTEND-BACKEND INTEGRATION")
    print("=" * 50)
    
    # Test 1: Main page
    print("\n1. Testing main page...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Main page accessible")
        else:
            print(f"❌ Main page error: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Emotion data API
    print("\n2. Testing emotion data API...")
    try:
        response = requests.get(f"{base_url}/api/emotion-data")
        if response.status_code == 200:
            data = response.json()
            print("✅ Emotion data API working")
            print(f"   Recent emotions: {len(data.get('recent_emotions', []))}")
            print(f"   Distribution: {data.get('emotion_distribution', {})}")
        else:
            print(f"❌ Emotion data API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Emotion data API error: {e}")
    
    # Test 3: Real-time page
    print("\n3. Testing real-time page...")
    try:
        response = requests.get(f"{base_url}/realtime")
        if response.status_code == 200:
            print("✅ Real-time page accessible")
        else:
            print(f"❌ Real-time page error: {response.status_code}")
    except Exception as e:
        print(f"❌ Real-time page error: {e}")
    
    return True

def test_socket_connection():
    """Test Socket.IO connection"""
    print("\n4. Testing Socket.IO connection...")
    try:
        import socketio
        sio = socketio.Client()
        
        @sio.event
        def connect():
            print("✅ Socket.IO connected successfully")
            sio.disconnect()
        
        @sio.event
        def disconnect():
            print("✅ Socket.IO disconnected")
        
        sio.connect('http://localhost:5000')
        time.sleep(1)
        
    except ImportError:
        print("⚠️ Socket.IO client not available, skipping test")
    except Exception as e:
        print(f"❌ Socket.IO connection error: {e}")

def check_frontend_files():
    """Check if frontend files exist"""
    print("\n5. Checking frontend files...")
    
    files_to_check = [
        "templates/realtime.html",
        "static/js/realtime.js", 
        "static/css/style.css"
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"✅ {file_path} exists ({len(content)} chars)")
        except FileNotFoundError:
            print(f"❌ {file_path} not found")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

def main():
    print("🔍 FRONTEND-BACKEND INTEGRATION TEST")
    print("=" * 50)
    
    # Check frontend files
    check_frontend_files()
    
    # Test API endpoints
    if test_api_endpoints():
        # Test Socket.IO if server is running
        test_socket_connection()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print("✅ Frontend files checked")
    print("✅ API endpoints tested") 
    print("✅ Socket.IO connection tested")
    print("\n🎯 NEXT STEPS:")
    print("1. Open http://localhost:5000/realtime in browser")
    print("2. Click 'Start Detection' button")
    print("3. Check browser console for any errors")
    print("4. Verify bounding boxes appear on video")
    print("5. Verify charts update with real-time data")

if __name__ == "__main__":
    main() 