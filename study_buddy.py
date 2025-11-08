# study_buddy.py

import streamlit as st
import os
import base64 
import io     
from rag_utils import (
    get_pdf_text, 
    get_text_chunks, 
    setup_vectorstore, 
    get_rag_response, 
    summarize_chapter,
    generate_exam_questions,
    evaluate_answer,
    detect_language
)

# --- Streamlit UI Setup ---

st.set_page_config(
    page_title="📚 Study Buddy AI Teacher",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Study Buddy AI Teacher")
st.caption("Powered by Google Gemini and Milvus RAG")

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "full_text" not in st.session_state:
    st.session_state.full_text = None
if "pdf_file_bytes" not in st.session_state: 
    st.session_state.pdf_file_bytes = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "pdf_language" not in st.session_state:
    st.session_state.pdf_language = "English"

# --- Sidebar for Configuration and Upload ---

with st.sidebar:
    st.header("1. Upload & Configure")
    
    # Class Selection
    class_name = st.selectbox(
        "Select your Class (for teacher persona)",
        ["Primary School English", "Middle School English", "High School English", "Primary School Kannada", "Middle School Kannada", "High School Kannada", "Primary School Hindi", "Middle School Hindi", "High School Hindi"],
        key="class_name"
    )
    
    # File Uploader
    pdf_file = st.file_uploader(
        "Upload your Chapter PDF (English, Hindi, or Kannada)",
        type="pdf",
        accept_multiple_files=False
    )
    
    # Processing Button
    if st.button("✨ Process Chapter"):
        if pdf_file is not None:
            # Clear previous state
            st.session_state.vectorstore = None
            st.session_state.chat_messages = []
            st.session_state.questions = []
            st.session_state.current_question_index = 0
            
            # --- CAPTURE FILE BYTES FOR VIEWER ---
            pdf_bytes = pdf_file.read()
            st.session_state.pdf_file_bytes = pdf_bytes
            pdf_file.seek(0) # Reset file pointer for text extraction
            
            with st.spinner("Step 1: Extracting text from PDF and detecting language..."):
                raw_text = get_pdf_text(pdf_file)
                st.session_state.full_text = raw_text
                
                detected_lang = detect_language(raw_text)
                st.session_state.pdf_language = detected_lang
                st.success(f"Text Extraction Complete! Chapter Language Detected: **{detected_lang}**")
                
            with st.spinner("Step 2: Chunking, Embedding, and Indexing in Milvus..."):
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = setup_vectorstore(text_chunks)
                st.success("RAG Indexing Complete! You can now chat.")
        else:
            st.error("Please upload a PDF file first.")

# --- Main Tabs ---
tab_chat, tab_summary, tab_exam = st.tabs(["💬 AI Teacher Chat", "📝 Summary & Translation", "🧠 Exam Mode"])

# --- Chat Tab ---
with tab_chat:
    
    # ----------------------------------------------------
    # PDF VIEWER SECTION (Top Half)
    # ----------------------------------------------------
    if st.session_state.pdf_file_bytes:
        st.subheader("📖 Chapter PDF Viewer")
        
        base64_pdf = base64.b64encode(st.session_state.pdf_file_bytes).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.divider() 

    # ----------------------------------------------------
    # AI TEACHER CHAT SECTION (Bottom Half)
    # ----------------------------------------------------
    st.subheader(f"💬 Chat with your {st.session_state.class_name} Teacher")
    
    if st.session_state.vectorstore is None:
        st.info("Please upload and process a PDF in the sidebar to start chatting.")
    else:
        # Display all chat history messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input and Processing
        if prompt := st.chat_input("Ask a question about the chapter..."):
            
            # 1. Store and Display User Message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 2. Process and Display Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    # Call the RAG function
                    response = get_rag_response(
                        question=prompt,
                        vectorstore=st.session_state.vectorstore,
                        class_name=st.session_state.class_name,
                        pdf_language=st.session_state.pdf_language 
                    )
                    
                    # Display the response 
                    st.markdown(response)
                    
                    # 3. Store Assistant Message for history persistence
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# --- Summary Tab (Always English, as requested) ---
with tab_summary:
    st.header("Chapter Summary ")
    
    if st.session_state.full_text:
        st.info(f"The original chapter language was detected as **{st.session_state.pdf_language}**.")
        if st.button("Generate Summary"):
            with st.spinner("Generating summary and ensuring translation to English..."):
                summary = summarize_chapter(
                    full_text=st.session_state.full_text,
                    class_name=st.session_state.class_name
                )
                st.subheader("📚 Chapter Summary")
                st.markdown(summary)
    else:
        st.info("Please upload and process a PDF in the sidebar to generate a summary.")

# --- Exam Tab ---
with tab_exam:
    st.header("Exam Time")
    
    if st.session_state.full_text is None:
        st.info("Please upload and process a PDF in the sidebar to start the exam.")
    else:
        # 1. Generate Questions Button
        if not st.session_state.questions and st.button("Generate Exam Questions"):
            with st.spinner("Generating Exam..."):
                st.session_state.questions = generate_exam_questions(
                    full_text=st.session_state.full_text,
                    class_name=st.session_state.class_name,
                    pdf_language=st.session_state.pdf_language 
                )
                st.session_state.current_question_index = 0
                st.success(f"Questions Ready! (In {st.session_state.pdf_language} or English)")
                
        # 2. Display Current Question
        if st.session_state.questions:
            q_index = st.session_state.current_question_index
            if q_index < len(st.session_state.questions):
                st.subheader(f"Question {q_index + 1} of {len(st.session_state.questions)}")
                current_question = st.session_state.questions[q_index]
                st.code(current_question, language='markdown')
                
                # Student Answer Input
                student_answer = st.text_area(
                    "Your Answer:", 
                    key=f"answer_{q_index}", 
                    height=200
                )
                
                col1, col2 = st.columns(2)
                
                # 3. Evaluate Button
                with col1:
                    if st.button("Evaluate Answer", key=f"eval_{q_index}"):
                        if student_answer.strip():
                            with st.spinner("Evaluating your answer and preparing feedback..."):
                                evaluation = evaluate_answer(
                                    question=current_question,
                                    student_answer=student_answer,
                                    class_name=st.session_state.class_name,
                                    full_text=st.session_state.full_text,
                                    pdf_language=st.session_state.pdf_language 
                                )
                                # Store evaluation for the PREVIOUS question (which was just answered)
                                st.session_state[f"feedback_{q_index}"] = evaluation
                                st.session_state.current_question_index += 1
                        else:
                            st.warning("Please provide an answer before evaluating.")

                # 4. Display Feedback
                feedback_index = q_index - 1
                if feedback_index >= 0 and f"feedback_{feedback_index}" in st.session_state:
                    st.subheader(f"📝 Feedback for Question {feedback_index + 1}")
                    st.markdown(st.session_state[f"feedback_{feedback_index}"])
                    st.divider()

            else:
                st.success("🎉 **Exam Complete!** Review your feedback above or refresh to start a new exam.")
