import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]


from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


# --------------------------------------------------------------
# Tool functions
# -------------------------------------------------------------
@tool
def tavily(query: str):
    """To search the web"""
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 2,
            "include_answer": True,
        },
    )
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code} - {response.text}"


@tool
def policy_search(query: str):
    """To search the internal database for the customer service related inquiries"""

    from typing import TypedDict, List
    from langgraph.graph import StateGraph, END, START
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    # --------------------------------------------------------------
    # Vector Store Setup
    # -------------------------------------------------------------
    # 1. Document Uploading
    # file_path = r"C:\Users\Abishai Winston\Downloads\Customer-Service-Policy-ND.pdf"
    # document = PyPDFLoader(file_path).load()
    # document

    # 2. Splitting
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 500, chunk_overlap = 50, add_start_index = True
    # )
    # splitted_text = text_splitter.split_documents(document)
    # len(splitted_text)

    # 3. Setting up & Storing in the Vector Database
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # create database in vector store
    # pc.create_index(name="ragagent1", dimension=3072, metric="cosine", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
    # initializing vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name="ragagent1", embedding=embeddings
    )
    # adding the documents
    # ragagent1_vs = vector_store.add_documents(documents=splitted_text)

    # --------------------------------------------------------------
    # Define the Graph State
    # -------------------------------------------------------------
    class RAGState(TypedDict):
        question: str
        context: str
        answer: str

    # --------------------------------------------------------------
    # Node 1: Retrieve Context
    # -------------------------------------------------------------
    def retrieve(state: RAGState):
        query_embedding = embeddings.embed_query(state["question"])
        results = vector_store.similarity_search_by_vector(query_embedding)
        context = "\n\n".join([doc.page_content for doc in results])
        state["context"] = context
        return state

    # --------------------------------------------------------------
    # Node 2: Generate Response
    # -------------------------------------------------------------
    def generate(state: RAGState):
        rag_llm = init_chat_model(model="gpt-4.1", model_provider="openai")
        rag_final_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant answering questions from official customer service policies.",
                ),
                (
                    "human",
                    "Answer the following based only on the given context.\n\nContext:\n{context}\n\nQuestion:\n{question}",
                ),
            ]
        )
        rag_final_prompt = rag_final_prompt_template.invoke(
            {"context": state["context"], "question": state["question"]}
        )
        rag_final_response = rag_llm.invoke(rag_final_prompt)
        state["answer"] = rag_final_response
        return state

    # --------------------------------------------------------------
    # Generate Graph
    # -------------------------------------------------------------
    graph = StateGraph(RAGState)

    graph.add_node("Retrieve", retrieve)
    graph.add_node("Generate", generate)

    graph.add_edge(START, "Retrieve")
    graph.add_edge("Retrieve", "Generate")
    graph.add_edge("Generate", END)

    rag_app = graph.compile()

    final_state = rag_app.invoke({"question": query})
    print(f"final state of the RAG app is {final_state}")
    return final_state["answer"]


# --------------------------------------------------------------
# Tool definitions
# -------------------------------------------------------------
tools = [tavily, policy_search]
# --------------------------------------------------------------
# Model & Agent Inititiation
# -------------------------------------------------------------
model = init_chat_model(model="gpt-4.1", model_provider="openai")
Messages = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{text}"),
    ]
)
agent_executor = create_react_agent(model, tools)
# --------------------------------------------------------------
# Agent Execution - 1
# -------------------------------------------------------------
customer_question = (
    "How should staff respond if a service animal becomes unruly or disruptive?"
)
prompt = Messages.invoke({"text": customer_question})
agentic_response = agent_executor.invoke(prompt)
# --------------------------------------------------------------
# Agentic Reponse - 1
# -------------------------------------------------------------
for message in agentic_response["messages"]:
    message.pretty_print()

# --------------------------------------------------------------
# Agent Execution - 2
# -------------------------------------------------------------
prompt = Messages.invoke({"text": "Whats the weather in kanyakumari?"})
agentic_response = agent_executor.invoke(prompt)
# --------------------------------------------------------------
# Agentic Reponse - 2
# -------------------------------------------------------------
for message in agentic_response["messages"]:
    message.pretty_print()
