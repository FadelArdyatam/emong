import google.generativeai as genai
import os

# Mengatur API key
genai.configure(api_key="AIzaSyAEGNoQjDSQQOCd8tCf2JrDJQNt6_6QFwE")

# Mengambil daftar model
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Nama: {model.name}")
        print(f"Deskripsi: {model.description}")
        print(f"Metode yang Didukung: {model.supported_generation_methods}")
        print("-" * 50)

if __name__ == "__main__":
    pass
