import os
import re
import uuid
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone as PineconeClient
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
p_api_key = os.getenv('PINECONE_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')

pc = PineconeClient(api_key=p_api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"
)
vector_store = PineconeVectorStore(index=pc.Index("sqlchatbot"), embedding=embeddings)
retriever = vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

pdf_paths = ["sql-reference.pdf", "sql-reference-2.pdf"]
documents = []

for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

MAX_CHARS = 1000
safe_chunks = []

for chunk in chunks:
    text = chunk.page_content
    if len(text) <= MAX_CHARS:
        safe_chunks.append(chunk)
    else:
        for i in range(0, len(text), MAX_CHARS):
            safe_chunks.append(Document(
                page_content=text[i:i+MAX_CHARS],
                metadata=chunk.metadata
            ))

def clean_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  
    text = re.sub(r'\*(.*?)\*', r'\1', text)      
    text = re.sub(r'```(.*?)```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    text = re.sub(r'^\s*[\*\-]\s+', '- ', text, flags=re.MULTILINE)
    text = text.replace('*', '')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

memory = MemorySaver()

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node("chat_model", call_model)
workflow.set_entry_point("chat_model")
workflow.add_edge(START, "chat_model")
workflow.add_edge("chat_model", END)
app_graph = workflow.compile(checkpointer=memory)


app = Flask(__name__)

session_tracker = set()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    contents = data.get("contents")

    if not contents:
        return "Missing chat contents", 400

    user_input = contents[-1]["parts"][0].get("text", "")

    results = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are an expert SQL developer and database analyst. Your task is to:
- Understand and respond accurately to both technical SQL questions and natural language queries.
- Provide clear definitions and explanations when the user asks conceptual questions.
- Generate efficient, correct SQL queries when the user asks for data retrieval or analysis.

Context:
{context}

Instructions:
1. If the user asks a conceptual question (e.g., definitions, types, syntax, best practices), provide a clear and concise explanation.
2. If the user asks for data from a database, refer to the provided schema and generate efficient SQL.
3. Format both explanations and SQL for clarity and readability.
4. If you display a result table, you must follow this formatting exactly:

-Format it as an ASCII table using +---+ style borders.
-Use | to separate columns.
-Align each value perfectly below its column header.
-Pad each cell with spaces so all columns have equal width across rows.
-Numbers must be right-aligned, text left-aligned, dates can be left or center-aligned.
-Top and bottom borders must match the full width.
-Now format the following result using the prettytable ASCII style with proper alignment.


User Question:
{user_input}

Answer (plain text, no markdown formatting):
"""

    session_id = data.get("session_id", str(uuid.uuid4()))
    is_new_session = session_id not in session_tracker

    session_tracker.add(session_id)

    if is_new_session:
        result = app_graph.invoke(
            {"messages": [
                SystemMessage(content=prompt),
                HumanMessage(content=user_input)
            ]},
            config={"configurable": {"thread_id": session_id}}
        )
    else:
        result = app_graph.invoke(
            {"messages": [
                HumanMessage(content=user_input)
            ]},
            config={"configurable": {"thread_id": session_id}}
        )

    ai_response = result["messages"]
    final_response = next((m.content for m in reversed(ai_response) if hasattr(m, "content")), "") if isinstance(ai_response, list) else getattr(ai_response, "content", str(ai_response))
    cleaned_response = clean_markdown(final_response)

    return cleaned_response

if __name__ == "__main__":
    app.run(debug=True, port=5050)
