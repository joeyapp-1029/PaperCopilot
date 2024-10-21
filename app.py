import argparse
import streamlit as st
from pdf_wrapper import PDFWrapper
from digest_paper import PaperDigest
from llm_wrapper import LLMWrapper
import os

# Set the environment variable
CACHE_DIR = "./cache/"
rerank = False
rewrite_query = True

def is_digested(uploaded_filename):
    '''
    Check if the uploaded PDF has already been processed and the digest is stored in the cache.
    Returns True if the digest exists, False otherwise.
    '''
    path = os.path.join(CACHE_DIR, f"paper_digest_{uploaded_filename}.pkl")
    return os.path.exists(path)

# 0. setup rerank model
if rerank:
    from sentence_transformers import CrossEncoder
    reranker_model = CrossEncoder(model_name="BAAI/bge-reranker-large", max_length=512)

# 1. Upload PDF

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

previous_qa = []

if uploaded_file:
    # 2. Parse the PDF
    with st.spinner('Extracting text from the uploaded PDF...'):
        uploaded_filename = uploaded_file.name
        cache_path = os.path.join(CACHE_DIR, f"paper_digest_{uploaded_filename}.pkl")
        # load pdf file
        pdf_wrapper = PDFWrapper.from_byte_stream(uploaded_file)
        # RAG Step 0 - Initialization
        paper_digest = PaperDigest(pdf_wrapper)
        st.success("File successfully digested and cached")

    st.success("Finished processing. Ask me anything about this doc!")


    
    # 3. RAG Step 1 - Input queries from user
    user_query = st.text_input("Enter your question:")

    if user_query:
        # 3.1. Query re-write
        if rewrite_query:
            query = LLMWrapper.rewrite_query(query=user_query)
            print(f"[DEBUG] User query: {user_query},\nRewritten query: {query}")
        # 4. Generate answer
        with st.spinner('Generating an answer...'):
            # RAG Step 1.1 - Retrieval
            docs = paper_digest.get_best_doc(query=user_query,top_k=10) # a list of str
            if rerank:
                docs = paper_digest.rerank_docs(
                    reranker_model=reranker_model,
                    query=query,
                    docs = docs,
                    top_k = 3,
                    output_score=False
                )
            print(f"[DEBUG] Retrieved doc for query: {query} - {docs}")
            # RAG Step 2 - Aguementation & Step 3 - Generation
            answer = LLMWrapper.ask(query=query,docs = docs)
            # answer = f"DEBUG |the answer for question {user_query} is ..."
            # generate potential questions
            next_question = LLMWrapper.generate_next_question(prev_query=user_query,
                                           prev_ans = answer,
                                           docs = docs)
            # next_question = '<DEBUG |next question>'
            

            # Check if 'previous_qa' exists in session_state, if not, create it
            if 'previous_qa' not in st.session_state:
                st.session_state.previous_qa = []
            
            # Store this Q&A to previous_qa in session_state
            st.session_state.previous_qa.append((user_query, answer))

        # 5. Show the answer
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("You may want to ask...")
        st.write(next_question)



    # Display previous questions and answers in the sidebar
    st.sidebar.title("Previous Questions & Answers")
    for question, answer in st.session_state.get('previous_qa', []):
        st.sidebar.markdown(f"**Q:** {question}")
        st.sidebar.markdown(f"**A:** {answer}")
        st.sidebar.markdown("---")
