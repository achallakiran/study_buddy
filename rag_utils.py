# rag_utils.py

import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langdetect import detect 

# LangChain Imports
from langchain_core.prompts import PromptTemplate 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus.vectorstores import Milvus 

# Milvus Imports
from pymilvus import MilvusClient

load_dotenv()

# --- Configuration ---
# Prioritize GEMINI_API_KEY, falling back to GOOGLE_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Set the environment variables explicitly for robustness
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY or "" 
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY or "" 

if not GEMINI_API_KEY:
    raise ValueError(
        "FATAL ERROR: GEMINI_API_KEY or GOOGLE_API_KEY is missing. "
        "Please run 'export GEMINI_API_KEY=...' in your terminal session."
    )

MODEL_CHAT = "gemini-2.5-flash"
MODEL_EMBEDDING = "models/text-embedding-004"
MILVUS_COLLECTION_NAME = "study_buddy_collection"
MILVUS_DB_PATH = "./milvus_study_buddy.db" # Milvus Lite local DB

# --- RAG Setup Functions ---

def get_pdf_text(pdf_file) -> str:
    """Extracts text from a single uploaded PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    # Use io.BytesIO(pdf_file.read()) if passing raw bytes, but since we reset the pointer, this is fine
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def detect_language(text: str) -> str:
    """Detects the primary language of the text."""
    try:
        lang_code = detect(text[:1000].replace('\n', ' '))
        if lang_code == 'hi':
            return 'Hindi'
        elif lang_code == 'kn':
            return 'Kannada'
        elif lang_code == 'en':
            return 'English'
        else:
            return 'English' 
    except:
        return 'English'

def get_text_chunks(text: str) -> list[str]:
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def setup_vectorstore(text_chunks: list[str]) -> Milvus:
    """Creates embeddings and loads them into Milvus Lite."""
    print("Initializing Milvus Client...")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    # 1. Initialize Embedding Model 
    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODEL_EMBEDDING,
        api_key=GEMINI_API_KEY 
    )
    
    # 2. Initialize Milvus Client 
    milvus_client = MilvusClient(uri=MILVUS_DB_PATH)
    
    if milvus_client.has_collection(collection_name=MILVUS_COLLECTION_NAME):
        milvus_client.drop_collection(collection_name=MILVUS_COLLECTION_NAME)
        print(f"Dropped old collection: {MILVUS_COLLECTION_NAME}")

    # 3. Create Vector Store via LangChain wrapper (FIXED for retrieval)
    vectorstore = Milvus.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"uri": MILVUS_DB_PATH},
        text_field="text" # CRITICAL FIX for retrieval
    )
    print("Vector Store setup complete.")
    return vectorstore

# --- LLM Interaction Functions (Bilingual Logic) ---

def get_rag_response(question: str, vectorstore: Milvus, class_name: str, pdf_language: str) -> str:
    """Retrieves context and gets an answer from Gemini, conditionally providing bilingual response."""
    
    # --- FIX 1: Basic Conversation Check (Bypass RAG for simple chat) ---
    lower_question = question.lower().strip()
    greetings = ["hi", "hello", "hey", "hlo", "what's up", "how are you", "how are you?"]
    if any(g in lower_question for g in greetings):
        return f"Hello! I am your AI Study Buddy. I can answer questions about the chapter you uploaded. What would you like to know?"
    
    # --- Check for Summary/Overview Questions ---
    overview_phrases = ["what is this book about", "what is this chapter about", "summarize this chapter", "give me an overview"]
    if any(phrase in lower_question for phrase in overview_phrases):
        return "That's a broad question! For a comprehensive overview, please go to the **📝 Summary & Translation** tab and click 'Generate Summary'."
    # --- END Logic Gates ---


    # Standard RAG logic starts here
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n---\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(model=MODEL_CHAT, api_key=GEMINI_API_KEY)
    
    if pdf_language in ['Hindi', 'Kannada']:
        bilingual_instruction = (
            f"Since the chapter is in {pdf_language}, your full response must be **bilingual**. "
            f"First, provide the full, natural response in {pdf_language} (the PDF's original language). "
            f"Second, provide a precise translation of that exact response into English. "
            f"Use a visible separator line (e.g., '---') between the two language versions."
        )
    else:
        bilingual_instruction = "The final response must be in English only."

    RAG_PROMPT = PromptTemplate.from_template(
        """You are the **{class_name} teacher**. Your goal is to answer the student's question accurately and helpfully, using information from the provided chapter context as well as your experience as an experienced **{class_name} teacher**.
        
        {bilingual_instruction}
        
        If the context does not contain the answer, state the required response in the necessary language(s): In this case provide the answer based on your experience as an experienced **{class_name} teacher**.
        
        **Chapter Context:**
        {context}
        
        **Student Question:**
        {question}
        
        **Teacher Response:**"""
    )

    chain = RAG_PROMPT | llm
    
    response = chain.invoke({
        "class_name": class_name, 
        "bilingual_instruction": bilingual_instruction,
        "context": context, 
        "question": question
    })
    return response.content

def summarize_chapter(full_text: str, class_name: str) -> str:
    """Summarizes the chapter content entirely in English, and in the local language. Summaries separated by a visible separator like a new line and >>"""
    llm = ChatGoogleGenerativeAI(model=MODEL_CHAT, api_key=GEMINI_API_KEY)
    
    SUMMARY_PROMPT = PromptTemplate.from_template(
        """You are an expert academic translator and summarizer.
        The student is in the **{class_name}** class.
        Your task is to provide a comprehensive, clear summary of the following chapter text.
        **CRITICAL RULE: The final summary must be in English followed by summary in the local language in which the chapter is written.**
        
        **Chapter Text:**
        {text}
        
        **English Summary:**"""
    )
    
    chain = SUMMARY_PROMPT | llm
    
    response = chain.invoke({"class_name": class_name, "text": full_text})
    return response.content

def generate_exam_questions(full_text: str, class_name: str, pdf_language: str) -> list[str]:
    """Generates 5 open-ended questions for exam mode."""
    llm = ChatGoogleGenerativeAI(model=MODEL_CHAT, api_key=GEMINI_API_KEY)
    
    output_lang = pdf_language if pdf_language in ['Hindi', 'Kannada'] else 'English'

    Q_GEN_PROMPT = PromptTemplate.from_template(
        """You are an experienced **{class_name} teacher**. Based ONLY on the following chapter text, generate exactly **3=5 in-depth, open-ended** questions suitable for an exam.
        Output the questions in a simple, numbered list in **{output_lang}** (the original chapter language) followed by its translation in English in brackets in italics with no extra conversational text.
        
        **Chapter Text:**
        {text}
        """
    )
    
    chain = Q_GEN_PROMPT | llm
    response = chain.invoke({"class_name": class_name, "text": full_text, "output_lang": output_lang})
    
    questions = [q.strip() for q in response.content.split('\n') if q.strip() and any(q.strip().startswith(f"{i}.") for i in range(1, 4))]
    return questions

def evaluate_answer(question: str, student_answer: str, class_name: str, full_text: str, pdf_language: str) -> str:
    """Evaluates the student's answer and provides feedback, using bilingual output if needed."""
    llm = ChatGoogleGenerativeAI(model=MODEL_CHAT, api_key=GEMINI_API_KEY)
    
    if pdf_language in ['Hindi', 'Kannada']:
        eval_instruction = (
            f"Provide the full evaluation in **{pdf_language}**, followed by a line separator (---), "
            f"and then provide the exact translation of the evaluation in English."
        )
    else:
        eval_instruction = "Provide the full evaluation and explanation in local language as well as in English."

    EVAL_PROMPT = PromptTemplate.from_template(
        """You are the **{class_name} teacher**. Your task is to **evaluate** the student's answer to the question below.
        
        1. Provide a score out of 10.
        2. Provide constructive feedback, explaining the correct points and any missing or incorrect information.
        3. Reference the chapter context implicitly to ensure accuracy.
        
        **{eval_instruction}**
        
        **Question:** {question}
        **Student Answer:** {student_answer}
        
        ---
        **Chapter Context (for reference):**
        {full_text}
        
        **Evaluation and Explanation:**"""
    )
    
    chain = EVAL_PROMPT | llm
    
    response = chain.invoke({
        "class_name": class_name, 
        "eval_instruction": eval_instruction,
        "question": question, 
        "student_answer": student_answer, 
        "full_text": full_text
    })
    return response.content
