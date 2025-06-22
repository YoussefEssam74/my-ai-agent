from typing import List
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json

# Load environment variables
load_dotenv()

# Define message class
class Message:
    def __init__(self, type, content):
        self.type = type
        self.content = content

# Define a Gemini LLM wrapper
class GeminiLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)  # ✅ Correct model initialization

    def __call__(self, messages):
        try:
            contents = []
            for msg in messages:
                if isinstance(msg, dict):
                    # from tool messages
                    content = msg["content"]
                else:
                    # from prompt formatted messages
                    content = msg.content
                contents.append(content)

            response = self.model.generate_content(
                contents,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=64,
                    max_output_tokens=8192
                )
            )
            return response.text

        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

# Define the response model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]
    speed_of_response_in_second: str

# Simulated web search tool
@tool
def web_search(message: str) -> str:
    """Simulates a web search for the given query."""
    return f"Web search results for '{message}': [Simulated data]"

# Tools dictionary
tools = {"web_search": web_search}

# Load API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Initialize LLM and parser
llm = GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-pro"))
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools. Available tools: {tool_names}.
If a tool is needed, include '[web_search: query]' in your response.
Wrap the output in this format and provide no other text:
{format_instructions}
"""
    ),
    ("human", "{message}")
]).partial(
    format_instructions=parser.get_format_instructions(),
    tool_names=", ".join(tools.keys())
)

# Main execution logic
def execute_agent(message):
    messages = prompt.format_messages(message=message)
    response = llm(messages)

    # Handle tool calls
    while "[web_search:" in response:
        start_idx = response.index("[web_search:") + 12
        end_idx = response.index("]", start_idx)
        tool_query = response[start_idx:end_idx]
        tool_result = tools["web_search"](tool_query)

        messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
        response = llm(messages)

    # Parse final response
    try:
        parsed_response = parser.parse(response)
        return parsed_response.json()
    except Exception as e:
        return f"Error parsing response: {str(e)}"

# Run
query = input("What can I help you research? ")
output = execute_agent(query)
print(output)
