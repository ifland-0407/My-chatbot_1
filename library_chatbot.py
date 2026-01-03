# -*- coding: utf-8 -*-
import os
import sys
import hashlib
import streamlit as st
from pathlib import Path

# -------------------------------------------------------------------
# ✅ sqlite3 호환 (Streamlit Cloud 등 일부 환경에서 Chroma가 sqlite3 빌드 이슈를 일으킬 때 대응)
#    - 반드시 Chroma/ChromaDB import "이전"에 실행되어야 합니다.
# -------------------------------------------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_chroma import Chroma


# -------------------------------------------------------------------
# ✅ API Key (Streamlit secrets 또는 환경변수에서만 읽기)
# -------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# -------------------------------------------------------------------
# ✅ 업로드 PDF 저장 + 해시
# -------------------------------------------------------------------
def save_uploaded_pdf_and_get_hash(uploaded_file) -> tuple[str, str]:
    data = uploaded_file.getbuffer()
    file_hash = hashlib.md5(data).hexdigest()
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)

    pdf_path = str(tmp_dir / f"{file_hash}_{uploaded_file.name}")
    with open(pdf_path, "wb") as f:
        f.write(data)
    return pdf_path, file_hash


def get_persist_dir(key: str) -> str:
    base = Path("./chroma_db")
    base.mkdir(exist_ok=True)
    return str(base / key)


# -------------------------------------------------------------------
# ✅ 캐시 함수들
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_split_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(_docs, persist_directory: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 기존 DB가 있으면 로드 시도
    if os.path.isdir(persist_directory) and any(os.scandir(persist_directory)):
        try:
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception:
            pass

    # 새로 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    split_docs = text_splitter.split_documents(_docs)

    return Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory,
    )


@st.cache_resource(show_spinner=False)
def initialize_chain(selected_model: str, pdf_path: str, persist_dir: str):
    """
    ✅ 반드시 Runnable(rag_chain)을 반환해야 함!
    """
    pages = load_and_split_pdf(pdf_path)
    vectorstore = build_or_load_vectorstore(pages, persist_dir)
    retriever = vectorstore.as_retriever()

    # 질문 재구성 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if "
        "needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 답변 프롬프트
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "대답은 한국어로 하고, 존댓말을 써주세요. 이모지도 적당히 사용해주세요.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# -------------------------------------------------------------------
# ✅ Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="PDF 기반 RAG 챗봇", page_icon="📚")
st.header("PDF 기반 RAG 챗봇 💬📚")

option = st.selectbox("Select GPT M
