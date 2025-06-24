# AI Agent Research Assistant

This project implements a multi-agent AI system for research assistance, inspired by CrewAI-style architecture. It supports:

  Features

* 🤖 Multi-agent architecture (starting with `ResearchAgent`, expandable)
* 🧠 Integration with Gemini LLM API
* 🔍 Tool support via `LangChain` (currently includes DuckDuckGo search)
* 🖍️ Terminal output coloring using `termcolor`
* 📝 JSON output saved to file with structured fields
* ⏱️ Response evaluation metrics (speed, clarity, accuracy, etc.)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/YoussefEssam74/my-ai-agent.git
cd my-ai-agent

   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file:

   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=models/gemini-1.5-flash
   ```

## Usage

Run the script with:

```bash
python v2.py
```

* Enter a research topic.
* The system will gather information, log tools used, and optionally save the output to `output.json`.

## Output Example

The output will be printed in the terminal with color and stored as structured JSON:

```json
{
    "topic": "...",
    "summary": "...",
    "sources": "https://...",
    "tools_used": ["web_search"]
}
