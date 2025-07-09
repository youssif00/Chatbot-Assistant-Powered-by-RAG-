import os
import google.generativeai as genai

# Configure the Gemini API with your API key
genai.configure(api_key="Your api from Gemini API")

# Keep a global model instance
model = genai.GenerativeModel("gemini-2.0-flash")

def call_llama(prompt: str, history: list[dict] | None = None) -> str:
  
    # Start a fresh chat session per call
    chat_session = model.start_chat(history=history or [])
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"[Gemini API ERROR] {e}", flush=True)
        return "Error: Could not generate response."