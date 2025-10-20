
# RAG + ReAct Agent Project

## Overview

This project demonstrates how to build an intelligent agent that combines **Retrieval-Augmented Generation (RAG)** and **ReAct reasoning** using the **LangChain** and **LangGraph** frameworks. It integrates with **OpenAI**, **Tavily**, and **Pinecone** to perform web search, internal document retrieval, and reasoning-based response generation.

The goal is to create a hybrid AI agent capable of answering both general and organization-specific questions. It uses external APIs for real-time search and a Pinecone vector store for internal knowledge retrieval from documents such as company policies or manuals.

## Objectives

The primary objectives of this project are:

1. To showcase a production-ready implementation of a **ReAct agent** that performs both reasoning and action-taking through tools.
2. To demonstrate how to integrate **retrieval-based knowledge systems** with LLMs using LangGraph and Pinecone.
3. To illustrate how to structure AI pipelines for complex enterprise use cases such as internal policy Q&A, document assistance, and web search integration.
4. To provide a foundation for extending agentic systems to multiple data sources and specialized domains.

## Technical Stack

**Core Frameworks**
- LangChain
- LangGraph
- OpenAI
- Pinecone
- Tavily API
- dotenv

**Programming Language**
- Python 3.10+

**Dependencies**
- langchain  
- langgraph  
- langchain-openai  
- langchain-pinecone  
- langchain-community  
- requests  
- python-dotenv  
- pinecone-client  

## System Architecture and Flow

1. **Tool Definition Layer**
   - Two tools are defined:
     - **Tavily Search Tool:** Fetches live web information.
     - **Policy Search Tool:** Performs RAG-based document Q&A from an internal Pinecone database.
   - Each tool includes docstrings that act as descriptions for the ReAct agent to select appropriate tools during reasoning.

2. **RAG Pipeline**
   - Documents are loaded and split using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
   - Embeddings are created with `OpenAIEmbeddings`.
   - Vectors are stored or retrieved from a Pinecone index named `ragagent1`.
   - The retrieval function performs similarity search to fetch relevant content.
   - The generation node builds a structured prompt and generates a final response using an OpenAI model.

3. **ReAct Agent Execution**
   - The ReAct agent (via LangGraph’s `create_react_agent`) coordinates reasoning steps.
   - The model decides when to call `tavily` or `policy_search` based on the query context.
   - After executing the appropriate tool, it synthesizes the final answer and returns it as an AI message.

4. **Example Queries**
   - Internal query: “How should staff respond if a service animal becomes unruly or disruptive?”
     - The model routes to the `policy_search` tool, retrieves the relevant section from the customer service policy PDF, and generates a response.
   - General query: “What’s the weather in Kanyakumari?”
     - The model routes to the `tavily` search tool and retrieves real-time data.

## Project Setup

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/RAG-ReAct-Agent.git
cd RAG-ReAct-Agent
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
```

### 4. Create Pinecone Index
```python
from pinecone import Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
pc.create_index(
    name="ragagent1",
    dimension=3072,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
)
```

### 5. Run Application
```bash
python src/main.py
```

## Execution Flow

1. The program initializes two tools: `tavily` and `policy_search`.
2. It loads environment variables, connects to Pinecone, and initializes the LLM.
3. A chat prompt is built dynamically from system and human messages.
4. The agent chooses a reasoning path:
   - If the query requires live data, it calls Tavily.
   - If it requires policy-based context, it calls the internal RAG function.
5. The reasoning path and messages are printed step-by-step in the console.

## Use Cases

1. **Enterprise Policy Q&A**
   - Internal teams can upload and search company policy documents.
2. **Knowledge Assistant for Customer Support**
   - Enables staff to query policy documents interactively.
3. **Hybrid AI Search System**
   - Combines web and internal search for complete responses.
4. **Training and Compliance Agents**
   - Employees can ask questions related to compliance and get answers from official manuals.

## Future Updates

1. Add a web-based frontend using FastAPI or Streamlit to visualize results.  
2. Integrate memory and long-term reasoning across multiple queries.  
3. Enable document uploading and automatic vectorization through APIs.  
4. Incorporate a local LLM fallback option using Ollama or vLLM.  
5. Add logging and monitoring to track reasoning paths and tool usage.  
6. Expand to support multi-modal inputs like voice or file uploads.  
7. Implement agent evaluation metrics such as context accuracy and latency.  

## Conclusion

The RAG + ReAct Agent demonstrates how modern AI frameworks like LangChain and LangGraph can be orchestrated to create intelligent, context-aware assistants that combine reasoning, retrieval, and external tool use. It serves as a foundational blueprint for building domain-specific assistants in customer service, enterprise support, or knowledge management environments.
