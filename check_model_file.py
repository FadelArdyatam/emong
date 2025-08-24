#!/usr/bin/env python3
"""
Check model file tanpa import heavy libraries
"""

import os
import struct

def check_model_file():
    """Check model file secara detail"""
    
    model_path = "models/yolo_emotion_detection_v8s.pt"
    
    print(f"🔍 Checking model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file tidak ditemukan: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"✅ Model file ditemukan")
    print(f"📊 File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    
    # Check file header
    try:
        with open(model_path, 'rb') as f:
            # Read first few bytes to check format
            header = f.read(16)
            print(f"📊 File header (hex): {header.hex()}")
            
            # Check if it's a valid PyTorch file
            if header.startswith(b'PK'):
                print("✅ File appears to be a valid PyTorch model (ZIP format)")
            elif header.startswith(b'\x80'):
                print("✅ File appears to be a valid PyTorch model (pickle format)")
            else:
                print("⚠️ File format unclear, might be corrupt")
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    return True

def check_directory():
    """Check models directory"""
    
    models_dir = "models"
    print(f"\n🔍 Checking models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory tidak ditemukan: {models_dir}")
        return False
    
    print(f"✅ Models directory ditemukan")
    
    # List all files
    files = os.listdir(models_dir)
    print(f"📁 Files in models directory:")
    
    for file in files:
        file_path = os.path.join(models_dir, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  📄 {file} - {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        else:
            print(f"  📁 {file} (directory)")
    
    return True

def main():
    """Main function"""
    
    print("🚀 Model File Diagnostic Tool")
    print("=" * 50)
    
    # Check models directory
    check_directory()
    
    # Check specific model file
    check_model_file()
    
    print("\n" + "=" * 50)
    print("💡 Recommendations:")
    print("1. If model file is too small (< 1MB), it might be corrupt")
    print("2. If file format is unclear, try re-downloading the model")
    print("3. Check if the model was trained properly for emotion detection")

if __name__ == "__main__":
    main() 