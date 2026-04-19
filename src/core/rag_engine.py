from typing import List, Tuple, Generator

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.config import LLM_CONFIG, RETRIEVER_CONFIG
from src.core.reranker import CrossEncoderReranker


def _get_prompt_template(user_input: str) -> PromptTemplate:
    """
    Trả về PromptTemplate phù hợp ngôn ngữ (tiếng Việt / English).

    Args:
        user_input: Câu hỏi của người dùng để phát hiện ngôn ngữ.

    Returns:
        PromptTemplate đã được chọn.
    """
    vietnamese_chars = (
        "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợ"
        "úùủũụưứừửữựýỳỷỹỵđ"
    )
    is_vietnamese = any(char in user_input.lower() for char in vietnamese_chars)

    if is_vietnamese:
        template = """Sử dụng ngữ cảnh và lịch sử trò chuyện dưới đây để trả lời câu hỏi.
Nếu bạn không biết, chỉ cần nói là bạn không biết.
Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.

Lịch sử trò chuyện gần đây:
{chat_history}

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời:"""
    else:
        template = """Use the following context and conversation history to answer the question.
If you don't know the answer, just say you don't know.
Keep your answer concise (3-4 sentences).

Recent Conversation History:
{chat_history}

Context: {context}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"],
    )


def run_rag_chain(
    user_question: str,
    vector_store,
    chat_history: str = "",
) -> Tuple[Generator, List[Document], List[str]]:
    """
    Thực thi toàn bộ RAG pipeline với metadata filtering và cross-encoder reranking.

    Pipeline:
        1. Retrieve candidates từ FAISS (có/không có metadata filter).
        2. Rerank candidates bằng CrossEncoderReranker.
        3. Xây dựng context từ top-k reranked docs.
        4. Stream câu trả lời từ LLM.

    Args:
        user_question: Câu hỏi của người dùng.
        vector_store: FAISS VectorStore đã được khởi tạo.
        chat_history: Chuỗi lịch sử trò chuyện (mặc định rỗng).

    Returns:
        Tuple gồm:
            - response_stream: Generator stream câu trả lời từ LLM.
            - reranked_docs: Danh sách Document sau khi rerank.
            - unique_sources: Danh sách tên file nguồn duy nhất.
    """

    # ------------------------------------------------------------------ #
    # 1. Khởi tạo LLM                                                     #
    # ------------------------------------------------------------------ #
    llm = OllamaLLM(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        top_p=LLM_CONFIG["top_p"],
        repeat_penalty=LLM_CONFIG["repeat_penalty"],
    )

    # ------------------------------------------------------------------ #
    # 2. Cấu hình Retriever (có hoặc không có metadata filter)            #
    # ------------------------------------------------------------------ #
    selected_docs: List[str] = st.session_state.get("selected_docs", [])

    if selected_docs:
        # Người dùng chọn một số file cụ thể → áp dụng metadata filter
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG["search_type"],
            search_kwargs={
                "k": 10,
                "filter": {"source": {"$in": selected_docs}},
            },
        )
    else:
        # Không có filter → tìm kiếm trên toàn bộ corpus
        retriever = vector_store.as_retriever(
            search_type=RETRIEVER_CONFIG["search_type"],
            search_kwargs={"k": 10},
        )

    # ------------------------------------------------------------------ #
    # 3. Retrieve candidate documents                                      #
    # ------------------------------------------------------------------ #
    candidate_docs: List[Document] = retriever.invoke(user_question)

    # ------------------------------------------------------------------ #
    # 4. Rerank bằng Cross-Encoder                                        #
    # ------------------------------------------------------------------ #
    reranker = CrossEncoderReranker()
    reranked_docs: List[Document] = reranker.rerank(
        query=user_question,
        documents=candidate_docs,
        top_k=3,
    )

    # ------------------------------------------------------------------ #
    # 5. Xây dựng context từ reranked docs                                #
    # ------------------------------------------------------------------ #
    context_text = "\n\n".join(doc.page_content for doc in reranked_docs)

    # Lấy danh sách tên file nguồn duy nhất (giữ thứ tự xuất hiện)
    seen: set = set()
    unique_sources: List[str] = []
    for doc in reranked_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            unique_sources.append(src)

    # ------------------------------------------------------------------ #
    # 6. Tạo prompt và stream câu trả lời                                 #
    # ------------------------------------------------------------------ #
    prompt = _get_prompt_template(user_question)

    chain = prompt | llm | StrOutputParser()

    response_stream = chain.stream(
        {
            "context": context_text,
            "question": user_question,
            "chat_history": chat_history,
        }
    )

    return response_stream, reranked_docs, unique_sources