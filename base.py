from typing import List
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from langchain_community.tools import DuckDuckGoSearchRun
import json
import time

# Load environment variables
load_dotenv()

# --------- Message Class ---------
class Message:
    def __init__(self, type, content):
        self.type = type
        self.content = content

# --------- Response Model ---------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: str
    tools_used: List[str]
    people: List[str]
    organizations: List[str]
    events: List[str]
    places: List[str]
    speed_of_response: str
    accuracy_of_response: str
    completeness_of_response: str
    relevance_of_response: str
    clarity_of_response: str
    creativity_of_response: str
    depth_of_response: str
    speed_of_response_in_second: str

# --------- Gemini LLM Wrapper ---------
class GeminiLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def __call__(self, messages):
        try:
            contents = []
            for msg in messages:
                if msg.type == "human":
                    role = "user"
                    content = msg.content
                elif msg.type == "system":
                    role = "user"  # Gemini doesn't support "system"
                    content = f"System instruction: {msg.content}"
                else:
                    continue
                contents.append({"role": role, "parts": [{"text": content}]})

            model = genai.GenerativeModel(model_name=self.model_name)
            response = model.generate_content(
                contents=contents,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=64,
                    max_output_tokens=8192
                )
            )
            return response.text

        except Exception as e:
            print("Gemini API error:", str(e))
            return json.dumps({
                "topic": "N/A",
                "summary": "Gemini API quota exceeded or error occurred.",
                "sources": [],
                "tools_used": [],
                "speed_of_response_in_second": "0"
            })

# --------- Search Tool (DuckDuckGo) ---------
search_tool = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    Performs a DuckDuckGo search and returns ONE highly relevant URL.
    """
    results = search_tool.invoke(query)

    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                title = item.get("title", "").lower()
                link = item.get("link", "")
                if "alexandria" in title and ("places" in title or "visit" in title or "weekend" in title):
                    return link
        for item in results:
            if isinstance(item, dict) and "link" in item:
                return item["link"]
    return "No relevant URL found."

tools = {"web_search": web_search}

# --------- Prompt Template ---------
llm = GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash"))
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools. Available tools: {tool_names}.
If a tool is needed, include '[tool_name: query]' in your response.
Wrap the output in this format, and for the 'sources' field, return only direct URLs (e.g., https://...):
{format_instructions}
"""
    ),
    ("human", "{message}")
]).partial(
    format_instructions=parser.get_format_instructions(),
    tool_names=", ".join(tools.keys())
)

# --------- Execution Logic ---------
def execute_agent(message):
    start = time.time()

    messages = prompt.format_messages(message=message)
    response = llm(messages)

    while "[web_search:" in response:
        start_idx = response.index("[web_search:") + 12
        end_idx = response.index("]", start_idx)
        tool_query = response[start_idx:end_idx].strip()
        tool_result = tools["web_search"](tool_query)

        messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
        response = llm(messages)

    try:
        parsed_response = parser.parse(response)
        parsed_response.tools_used = ["web_search"]
        parsed_response.speed_of_response_in_second = str(round(time.time() - start, 2))
        return parsed_response.model_dump_json()
    except Exception as e:
        return f"Error parsing response: {str(e)}"

# --------- Main Entry ---------
if __name__ == "__main__":
    query = input("What can I help you research? ")
    output = execute_agent(query)
    print(output)
