import streamlit as st
from src.core.router import classify_intent
from src.engines.chat_engine import run_chat
from src.engines.summarize_engine import run_summarize
from src.engines.rag_engine import run_rag_chain


def execute_query(
    query, history_str, vector_store, bm25_retriever, session_id, selected_files
):
    """Phân loại và điều phối chính xác đến Engine cần thiết"""

    # 1. Gọi Router
    intent = classify_intent(query, history_str)

    # Ghi log intent ra UI
    st.caption(f"🧠 **System Intent Router:** Kích hoạt luồng `[{intent}]`")

    meta_info = {"intent": intent, "is_relevant": True, "used_web": False}
    source_docs = []

    # 2. Điều phối bằng Match-Case (Tốc độ Cực hạn)
    match intent:
        case "CHAT":
            response_stream = run_chat(query, history_str)

        case "SUMMARIZE":
            response_stream = run_summarize(query, selected_files)

        case "WEB" | "RAG" | "EXPLAIN" | "COMPARE" | "EXTRACT" | "TRANSLATE":
            # Tương lai bạn tạo thêm Engine thì tách case ra.
            # Hiện tại cứ quăng vào con quái vật RAG Engine xử lý hết các vụ tìm kiếm/phân tích.
            response_stream, source_docs, rag_meta = run_rag_chain(
                user_question=query,
                vector_store=vector_store,
                bm25_retriever=bm25_retriever,
                session_id=session_id,
                chat_history=history_str,
                selected_files=selected_files,
            )
            meta_info.update(rag_meta)

        case _:
            response_stream = run_chat(query, history_str)  # Fallback

    return response_stream, source_docs, meta_info
