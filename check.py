import google.generativeai as genai
import os

# Mengatur API key
genai.configure(api_key="AIzaSyDgU8o7SLf3iVLdjkFQ3ZOpvij7Isi1lAk")

# Mengambil daftar model
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Nama: {model.name}")
        print(f"Deskripsi: {model.description}")
        print(f"Metode yang Didukung: {model.supported_generation_methods}")
        print("-" * 50)

if __name__ == "__main__":
    pass
