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
# ✅ 유틸: 업로드 PDF 저장 + 해시 만들기
#    - 같은 파일명이라도 내용이 다르면 다른 DB를 쓰게 하려고 해시 사용
# -------------------------------------------------------------------
def save_uploaded_pdf_and_get_hash(uploaded_file) -> tuple[str, str]:
    data = uploaded_file.getbuffer()
    file_hash = hashlib.md5(data).hexdigest()  # 간단/충분
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = str(tmp_dir / f"{file_hash}_{uploaded_file.name}")
    with open(pdf_path, "wb") as f:
        f.write(data)
    return pdf_path, file_hash


def get_persist_dir(file_hash_or_name: str) -> str:
    base = Path("./chroma_db")
    base.mkdir(exist_ok=True)
    return str(base / file_hash_or_name)


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
            # 손상/버전불일치 등의 이유로 로드 실패하면 새로 생성
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
    # 1) PDF -> pages
    pages = load_and_split_pdf(pdf_path)

    # 2) Vector DB
    vectorstore = build_or_load_vectorstore(pages, persist_dir)
    retriever = vectorstore.as_retriever()

    # 3) 질문 재구성 프롬프트
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

    # 4) QA 프롬프트
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer perfect. please use emoji with the answer. "
        "대답은 한국어로 하고, 존댓말을 써줘.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 5) RAG 체인 구성
    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# -------------------------------------------------------------------
# ✅ Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="국립부경대 도서관 규정 Q&A", page_icon="📚")
st.header("국립부경대 도서관 규정 Q&A 챗봇 💬📚")

# 모델 선택
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))

# PDF 선택: (1) 레포에 있는 기본 PDF 경로, (2) 업로드
DEFAULT_PDF = "[챗봇프로그램및실습] 부경대학교 규정집.pdf"

uploaded = st.file_uploader("PDF를 업로드하거나, 기본 PDF로 실행하세요.", type=["pdf"])

pdf_path = None
persist_dir = None

if uploaded is not None:
    pdf_path, file_hash = save_uploaded_pdf_and_get_hash(uploaded)
    persist_dir = get_persist_dir(file_hash)
else:
    if os.path.exists(DEFAULT_PDF):
        pdf_path = DEFAULT_PDF
        # 기본 PDF는 파일명(stem) 기준으로 persist_dir 생성
        persist_dir = get_persist_dir(Path(DEFAULT_PDF).stem)

if not pdf_path or not persist_dir:
    st.info("먼저 PDF를 업로드하시거나, 레포에 기본 PDF 파일을 추가해주세요.")
    st.stop()

# ✅ 여기서 반드시 rag_chain이 반환되어야 함 (이게 기존 오류의 핵심)
rag_chain = initialize_chain(option, pdf_path, persist_dir)

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 기존 대화 렌더링
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 입력
if prompt_message := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)

            answer = response.get("answer", "")
            st.write(answer)

            with st.expander("참고 문서 확인"):
                for doc in response.get("context", []):
                    src = doc.metadata.get("source", "source")
                    st.markdown(src, help=doc.page_content)
