# 🧠 AI Agent Research Assistant

This project implements a multi-agent AI system for research assistance, inspired by CrewAI-style architecture.

## 🚀 Features

- 🤖 Multi-agent architecture (starting with `ResearchAgent`, expandable)
- 🧠 Integration with Gemini LLM API
- 🔍 Tool support via LangChain (currently includes DuckDuckGo search)
- 🎨 Colored terminal output using `termcolor`
- 📝 Structured JSON output saved to file
- ⏱️ Response evaluation metrics (speed, clarity, accuracy, etc.)

## ⚙️ Setup

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

## 🧪 Usage

Run the assistant using:

```bash
python v2.py
```

- Enter a research topic.
- The agent will gather information, log tools used, and optionally save output to `output.json`.

## 📄 Output Example

The output will be:
- Printed in the terminal (with colors)
- Saved in a structured `output.json` file

```json
{
  "topic": "Your research topic here",
  "summary": "Brief summary of findings...",
  "sources": "https://example.com",
  "tools_used": ["web_search"]
}
```
