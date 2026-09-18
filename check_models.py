import os
import google.generativeai as genai

# Apni API key yahan daal (ya .env se load kar)
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

print("Available Models for your API Key:")
for m in genai.list_models():
    # Sirf wo models filter out kar rahe hain jo text generate (generateContent) kar sakte hain
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")