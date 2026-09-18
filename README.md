# ResearchGPT 🔬
**Autonomous Multi-Agent Intelligence Engine**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Production-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.8_Flash-blue)

## Overview

**ResearchGPT** is an enterprise-grade, multi-agent AI research system designed to autonomously synthesize academic-quality reports. Bypassing the limitations of superficial web searches and single-shot LLM prompts, ResearchGPT employs a sophisticated, cyclic LangGraph orchestration pipeline. It autonomously plans sub-queries, filters domains, extracts dense factual context, rigorously grades its own findings against Ragas-style metrics, and dynamically self-corrects until its strict quality thresholds are met. 

The result is a deeply analytical, heavily cited, and structurally flawless Markdown report generated exclusively from high-trust sources.

## System Architecture

ResearchGPT relies on a multi-agent state machine orchestrated by LangGraph, with robust state persistence in PostgreSQL and vector storage in Qdrant.

```mermaid
flowchart TD
    %% Define Styles
    classDef user fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    classDef api fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    classDef agent fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    classDef db fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff
    classDef external fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff

    User[User Input (Streamlit UI)]:::user --> API[FastAPI Backend]:::api
    API --> State[LangGraph State]:::api
    API --> Postgres[(PostgreSQL Task Management)]:::db
    
    State --> Planner[Planner Node]:::agent
    Planner --> Search[Search Node]:::agent
    Search <--> Providers[Tavily / DuckDuckGo]:::external
    Search --> Retriever[Retriever Node]:::agent
    Retriever <--> Qdrant[(Qdrant Vector DB)]:::db
    Retriever --> Critic[Critic Node]:::agent
    
    Critic -->|Score < 0.75| Optimizer[Query Optimizer Node]:::agent
    Optimizer --> Search
    
    Critic -->|Score >= 0.75| Writer[Writer Node]:::agent
    Writer --> FinalReport[Structured Markdown Report]:::api
    Writer --> API
```

## Core Features

*   **Multi-Agent Orchestration (LangGraph):** A deeply cyclic state machine (Planner → Search → Retriever → Critic → Optimizer/Writer) ensuring high-fidelity research through iterative self-correction.
*   **Trust-Tier Domain Filtering:** Aggressive up-stream domain filtering guarantees low-quality homework sites (e.g., Brainly, Quora) are explicitly blocked, pulling context exclusively from academic, policy, and journalistic domains.
*   **Multi-Key API Rotation:** Natively routes around free-tier `429 RESOURCE_EXHAUSTED` quota limits by seamlessly load-balancing requests across an array of fallback Google API keys.
*   **SmartRetry Mechanics:** Custom tenacity decorators protect the pipeline from temporary `503/504` server spikes, executing targeted 3-second wait/retries before failover without interrupting the larger workflow.
*   **Graceful Degradation:** The pipeline tracks iteration cycles and gracefully degrades to partial synthesis if maximum research depth is exhausted, preventing infinite loops and hallucinations.
*   **Native PDF Export:** The Streamlit frontend supports single-click compilation of Markdown synthesis into highly formatted PDF deliverables.

## Tech Stack

*   **Backend Interface:** FastAPI, SQLAlchemy, PostgreSQL
*   **AI & Orchestration:** LangGraph, LangChain, Google Gemini API (gemini-3.8-flash)
*   **Search & Vectorization:** Tavily Search API, DuckDuckGo (Fallback), Qdrant Vector Database
*   **Frontend UI:** Streamlit, fpdf2, markdown

## Setup & Installation

**1. Clone & Install Dependencies:**
```bash
git clone https://github.com/your-org/ResearchGPT.git
cd ResearchGPT
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

**2. Configure Environment Variables:**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEYS="your_key_1,your_key_2"
TAVILY_API_KEY="your_tavily_key"
DATABASE_URL="postgresql://user:password@localhost/deeptrace"
```

**3. Initialize the Backend Services:**
Ensure PostgreSQL is running locally, then launch the FastAPI server:
```bash
uvicorn app.api.main:app --reload --port 8000
```

**4. Launch the Enterprise Interface:**
Open a new terminal window, activate the virtual environment, and run:
```bash
streamlit run app/frontend/app.py
```