from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import List
import google.generativeai as genai

# Load environment variables
load_dotenv()
print('GEMINI KEY: ', os.getenv("GEMINI_API_KEY"))
print("GEMINI MODEL NAME: ", os.getenv("GEMINI_MODEL"))

def llm_gemini(question):
    # Proper Gemini setup
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-pro"))

    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

# Define the response model (optional - not used yet)
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]

# Run the chatbot
inputx = input("Enter your question: ")
print(llm_gemini(inputx))
