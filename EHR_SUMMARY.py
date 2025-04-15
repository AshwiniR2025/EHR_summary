import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time
import warnings

# Load environment variables
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize the LLM


llm = ChatGroq(model="llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a medical report analyzer with great knowledge in understanding lab results and health data.
    Summarize or answer the following based on the medical report.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

st.title("🧙 Abracadabra: Medical Report Analyzer")

uploaded_file = st.file_uploader("📄 Upload your medical report (PDF)", type=['pdf'])

summary_prompt = ChatPromptTemplate.from_template(
    """
    You are a medical expert. Please summarize the following medical report with a focus on key test results, conditions, and doctor recommendations:

    <context>
    {context}
    </context>
    """
)

def create_vector_embedding_and_summary(file):
    with st.spinner("🔄 Processing and embedding document..."):
        with open(file.name, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(file_path=file.name)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(split_docs, embeddings)

        st.session_state.vectors = vectordb
        st.session_state.docs = split_docs

        # Auto-summary
        summary_chain = create_stuff_documents_chain(llm, summary_prompt)
        summary_response = summary_chain.invoke({"context": split_docs})
        # If response is a string (not dict), just assign it directly
        st.session_state.summary = summary_response if isinstance(summary_response, str) else summary_response.get("answer", "")


        st.success("✅ Document processed and summarized!")

if uploaded_file and "vectors" not in st.session_state:
    create_vector_embedding_and_summary(uploaded_file)

if "summary" in st.session_state:
    st.subheader("📝 Medical Report Summary")
    st.write(st.session_state.summary)

user_prompt = st.text_input("🩺 Ask a question about your medical report")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please upload a report to ask questions.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write("🧠 **Answer:**", response['answer'])
        st.caption(f"⏱️ Response time: {time.process_time() - start:.2f}s")

       




