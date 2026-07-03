from google import genai
import os
import io
import sys
from dotenv import load_dotenv

# Fix Windows Unicode issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Initialize the Gemini client
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
client = genai.Client(api_key=API_KEY)



def process_response(response: str) -> str:
   
    prompt = f"""
    You are a helpful assistant. Extract concise, meaningful information from the following response.
    The summary should be short(strictlyless than 10 words).

    Response:
    {response}

    Summary:
    """

    try:
        # Use the gemma model to generate the summary
        result = client.models.generate_content(
            model="gemma-3-4b-it",
            contents=prompt
        )
        summary = result.text.strip()
        return summary

    except Exception as e:
        #print(f"Error processing response: {e}")
        return ""