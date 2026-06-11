```text
 ____                 _____                     
|  _ \  ___  ___ _ __|_   _| __ __ _  ___ ___ 
| | | |/ _ \/ _ \ '_ \ | || '__/ _` |/ __/ _ \
| |_| |  __/  __/ |_) || || | | (_| | (_|  __/
|____/ \___|\___| .__/ |_||_|  \__,_|\___\___|
                |_|                           
```

# DeepTrace
**LangGraph-powered Multi-Agent Research System**

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.2.0-darkgreen?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red?style=for-the-badge)

## What is DeepTrace?
DeepTrace is a fully autonomous AI research assistant. Given a topic, it orchestrates multiple specialized AI agents to break down the query, scrape the web, build a local vector database, evaluate the quality of its own research, self-correct if needed, and synthesize a hallucination-free markdown report with citations.

## Current Architecture

```text
[START]
   ↓
[Planner]     <-- Breaks query into sub-questions
   ↓
[Search] <─────────────────────────────┐
   ↓                                   │
[Qdrant(RAG)] <-- Indexes embeddings   │ (loops if quality<7)
   ↓                                   │
[Retriever]   <-- Extracts Top-K       │
   ↓                                   │
[Critic]      <-- Evaluates Quality ───┘
   │               
   ├── [Decision: APPROVED]
   ↓
[Writer]      <-- Synthesizes final report
   ↓
[Report]
```

## Tech Stack
| Component | Technology | Purpose |
| --- | --- | --- |
| **Orchestration** | LangGraph | Manages agent states and cyclic conditional routing |
| **Agents** | LangChain Core | Constructs LLM chains and structured outputs |
| **LLMs** | OpenRouter (Mistral/Llama) | Provides intelligent reasoning and synthesis |
| **Search Engine** | DuckDuckGo API | Fetches live, up-to-date web data |
| **Vector DB** | Qdrant | Stores embeddings for semantic search |
| **Embeddings** | FastEmbed (BGE-Small) | Generates local, free, CPU-bound vectors |
| **Validation** | Pydantic | Enforces strict JSON schemas for LLM outputs |

## Project Status

✅ **Phase 1: LangGraph Pipeline (Planner+Search+Writer)**
✅ **Phase 2: RAG + Qdrant Vector DB**
✅ **Phase 3: Critic Agent + Conditional Routing**
🔄 **Phase 4-8: Coming soon**

## How to Run Right Now

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key:**
   Create a `.env` file in the root directory and add your OpenRouter API key:
   ```text
   OPENROUTER_API_KEY="your_api_key_here"
   ```

3. **Execute Pipeline:**
   ```bash
   python main.py
   ```

## Expected Output
When you run the script, you will see real-time streaming logs as LangGraph moves between the Planner, Search, Retriever, and Critic nodes. You will see the Critic score the research and potentially trigger a retry loop (`--- CYCLE #2 ---`) if the score is below the threshold. The execution ends with a statistical summary (Search Loops, Sources Found, Docs Retrieved, Final Quality) followed by a formatted markdown report with inline citations based strictly on the retrieved context.

---
**GitHub:** [github.com/rajatrgupta/DeepTrace](https://github.com/rajatrgupta/DeepTrace)