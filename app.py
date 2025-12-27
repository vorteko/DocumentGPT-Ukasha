import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

from llama_index.llms.google_genai import GoogleGemini
import gradio as gr

global embed_model, db_client, collection_name, llm, current_index, current_query_engine
current_index = None
current_query_engine = None

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding model '{embed_model.model_name}' loaded successfully.\n")


CHROMA_PATH = "chroma_db_data"
print(f"Initializing ChromaDB at path: {CHROMA_PATH}")
db_client = PersistentClient(path=CHROMA_PATH)
print("ChromaDB client initialized.")

collection_name = "my_notes_collection"
print(f"ChromaDB collection name set to: {collection_name}.\n")


print("Initializing Google Gemini LLM...")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in Hugging Face secrets.")

llm = GoogleGemini(
    model="gemini-2.5-flash",  
    api_key=https://docs.google.com/forms/d/e/1FAIpQLSe9FKpDOwOx2pPZERGj-7trmooVi6tJ2c2wEaPYG9rYtmcrzA/formResponse,
    system_prompt=(
        "You are a highly professional, polite, and helpful AI assistant designed to answer questions "
        "only based on the provided document context. "
        "Your responses should be concise, clear, and directly address the user's query. "
        "If the answer is not found in the document, politely state that you cannot find the information "
        "in the provided text. Do not invent information. "
        "Maintain a friendly and encouraging tone. "
        "Use simple English, easy to understand for a general audience, as per Pakistani standards for clarity. "
        "Always start your very first response after a document upload with a warm greeting relevant to the document, "
        "e.g., 'Hello! I've processed your document. How can I help you today?'"
    )
)
print("Google Gemini LLM loaded successfully.\n")


def process_file_and_initialize_index(file_obj):
    global current_index, current_query_engine

    print("\n--- Document Upload Detected ---")
    try:
        db_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"Collection delete skipped: {e}")

    chroma_collection = db_client.get_or_create_collection(collection_name)
    print("ChromaDB collection ready.\n")

    if file_obj is None:
        return "Please upload a document to begin.", []

    if not isinstance(file_obj, list):
        file_paths = [file_obj.name]
    else:
        file_paths = [f.name for f in file_obj]

    print(f"Loading files: {file_paths}")
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    print("Creating vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        vector_store=chroma_collection,
    )
    current_index = index
    current_query_engine = current_index.as_query_engine(llm=llm)
    print("Index and query engine initialized.\n")

    return [[None, "Hello! I've successfully processed your document(s). How can I assist you today?"]], []

def chat_with_document(message, chat_history):
    global current_index, current_query_engine

    if current_index is None or not current_index.storage_context.docstore.docs:
        return "Please upload a document first using the 'Upload your document(s)' component."

    print(f"User: {message}")
    response = current_query_engine.query(message)
    print(f"Gemini: {response}\n")
    return str(response)


print("Launching Gradio UI...")
with gr.Blocks(theme=gr.themes.Soft(), title="Document GPT by Muhammad Ukasha Ghufran") as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #4B0082; font-size: 2.5em;">
            📚 My Document GPT <br>
            <span style="font-size: 0.7em; color: #6A5ACD;">Powered by Google Gemini & ChromaDB</span>
        </h1>
        <p style="text-align: center; color: #555; font-size: 1.1em;">
            Developed by <strong style="color: #4B0082;">Muhammad Ukasha Ghufran</strong>
        </p>
        <p style="text-align: center; color: #777;">
            Upload your document(s) (TXT, PDF, DOCX) and ask questions about their content.
        </p>
        """
    )

    file_input = gr.File(label="Upload your document(s)", file_count="multiple", type="filepath",
                         file_types=[".txt", ".pdf", ".docx"])
    gr.Markdown("<p style='text-align: center; color: #888; font-size: 0.9em;'>Max 50MB per file recommended.</p>")

    chat_interface = gr.ChatInterface(
        fn=chat_with_document,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask your question here...", container=False, scale=7),
        title="Chat with your Document",
        description="Ask any question from the uploaded documents.",
        theme="soft",
    )

    file_input.change(
        fn=process_file_and_initialize_index,
        inputs=[file_input],
        outputs=[chat_interface.chatbot, chat_interface.textbox]
    )

    gr.Markdown(
        """
        <hr>
        <p style="text-align: center; color: #999;">
            Built by <strong>Muhammad Ukasha Ghufran</strong>
        </p>
        """
    )

demo.launch()

