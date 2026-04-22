from typing import List, Tuple, Generator, Optional, Any

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.config import LLM_CONFIG, RETRIEVER_CONFIG


def setup_hybrid_retriever(
    vector_store: Any,
    documents: List[Document],
    search_filter: Optional[List[str]] = None,
) -> EnsembleRetriever:
    """
    Thiết lập bộ truy xuất lai (Hybrid Retriever) kết hợp 2 sức mạnh:
    1. Tìm kiếm từ khóa chính xác (BM25).
    2. Tìm kiếm theo ngữ nghĩa/ý nghĩa câu (FAISS Vector).

    Args:
        vector_store: Đối tượng FAISS chứa các vector đã được nhúng.
        documents: Danh sách toàn bộ tài liệu gốc (để nạp vào BM25).
        search_filter: Danh sách tên file để giới hạn phạm vi tìm kiếm.

    Returns:
        EnsembleRetriever: Bộ truy xuất kết hợp.
    """
    # 1. Lọc tài liệu cho thuật toán tìm kiếm từ khóa (BM25)
    filtered_docs = documents
    if search_filter:
        filtered_docs = [
            d for d in documents if d.metadata.get("source") in search_filter
        ]

    bm25_retriever = BM25Retriever.from_documents(filtered_docs)
    bm25_retriever.k = 3  # Lấy top 3 đoạn văn bản khớp từ khóa nhất

    # 2. Lọc tài liệu cho thuật toán tìm kiếm ngữ nghĩa (FAISS Vector DB)
    search_kwargs = {"k": 3}  # Lấy top 3 đoạn văn bản có vector gần nhất
    if search_filter:
        search_kwargs["filter"] = (
            lambda metadata: metadata.get("source") in search_filter
        )

    vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # 3. Kết hợp bằng EnsembleRetriever (Tỷ lệ trọng số 50-50 cho cả 2 phương pháp)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def run_rag_chain(
    user_question: str,
    vector_store: Any,
    documents: List[Document],
    chat_history: str = "",
    search_filter: Optional[List[str]] = None,
) -> Tuple[Generator, List[Document]]:
    """
    Thực thi luồng RAG tiêu chuẩn (Dành cho Cột Standard RAG).
    Bao gồm: Lấy ngữ cảnh -> Nạp vào Prompt -> Gọi LLM -> Trả về luồng chữ.

    Args:
        user_question: Câu hỏi của người dùng nhập vào.
        vector_store: Cơ sở dữ liệu Vector (FAISS).
        documents: Danh sách tài liệu gốc để lấy nội dung.
        chat_history: Chuỗi lịch sử trò chuyện để AI hiểu các câu hỏi trước đó.
        search_filter: Bộ lọc tài liệu từ Sidebar.

    Returns:
        Tuple chứa 2 phần tử:
        - Generator: Luồng ký tự trả về từ LLM (để tạo hiệu ứng gõ phím trên UI).
        - List[Document]: Danh sách các đoạn tài liệu được dùng làm trích dẫn.
    """
    # 1. Khởi tạo LLM với các thông số cấu hình chuẩn
    llm = OllamaLLM(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        top_p=LLM_CONFIG["top_p"],
        repeat_penalty=LLM_CONFIG["repeat_penalty"],
    )

    # 2. Logic tạo Prompt tự động nhận diện ngôn ngữ
    def get_prompt_template(user_input: str) -> PromptTemplate:
        # Bộ từ khóa nhận diện tiếng Việt
        vietnamese_chars = (
            "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
        )
        is_vietnamese = any(char in user_input.lower() for char in vietnamese_chars)

        # Trả về template tùy thuộc vào việc người dùng dùng ngôn ngữ nào
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
            Keep answer concise (3-4 sentences).
            
            Recent Conversation History:
            {chat_history}
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""

        return PromptTemplate(
            template=template, input_variables=["context", "question", "chat_history"]
        )

    # Gắn prompt
    prompt = get_prompt_template(user_question)

    # 3. Kích hoạt Retriever để đi săn tài liệu
    retriever = setup_hybrid_retriever(
        vector_store, documents=documents, search_filter=search_filter
    )

    # Lấy ra danh sách các đoạn text chứa câu trả lời (kèm metadata)
    source_docs = retriever.invoke(user_question)

    # Nối các đoạn text lại với nhau tạo thành một khối ngữ cảnh (Context)
    context_text = "\n\n".join(doc.page_content for doc in source_docs)

    # 4. Khởi tạo Pipeline (Chuỗi xử lý): Prompt -> LLM -> Lọc Output thành Text
    chain = prompt | llm | StrOutputParser()

    # 5. Kích hoạt luồng chạy sinh chữ (Streaming)
    response_stream = chain.stream(
        {
            "context": context_text,
            "question": user_question,
            "chat_history": chat_history,
        }
    )

    return response_stream, source_docs
