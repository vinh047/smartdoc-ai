"""
SmartDoc AI — Main Streamlit Application
=========================================
Advanced RAG pipeline với:
  • Multi-document upload & single FAISS vector store
  • Metadata filtering theo file (sidebar multiselect)
  • Cross-Encoder reranking
  • Persistent chat history (SQLite)
"""

import os
import tempfile
from typing import List

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import logic xử lý lõi
from src.core.document_processor import process_document
from src.core.rag_engine import run_rag_chain

# Import các hàm tương tác Database
from src.core.database import (
    init_db,
    create_new_session,
    get_all_sessions,
    add_message,
    get_messages_by_session,
)

# Import các component UI
from src.ui import (
    render_sidebar,
    render_header,
    render_file_uploader,
)

from src.config import EMBEDDING_CONFIG

# ─────────────────────────────────────────────
#  Page config (must be FIRST Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# 1. Khởi tạo Database (Chỉ chạy 1 lần để tạo file .db nếu chưa có)
init_db()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình Embedding…")
def load_embedder() -> HuggingFaceEmbeddings:
    """Khởi tạo embedder một lần duy nhất và cache lại."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={"device": EMBEDDING_CONFIG["device"]},
        encode_kwargs={"normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"]},
    )


# ─────────────────────────────────────────────
#  2. Khởi tạo các State cần thiết
# ─────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
# NEW: lưu tên các file đã upload và lựa chọn lọc
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames: List[str] = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs: List[str] = []


# ─────────────────────────────────────────────
#  3. Khu vực Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    render_sidebar()

    st.divider()
    st.markdown("### 💬 Lịch sử trò chuyện")

    # Nút tạo Phiên trò chuyện mới
    if st.button("➕ Cuộc trò chuyện mới", type="primary", use_container_width=True):
        new_id = create_new_session("Phiên chat mới")
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Các phiên gần đây:**")
    sessions = get_all_sessions()
    for sess in sessions:
        btn_label = f"📝 {sess['title']} ({sess['created_at'][:10]})"
        if st.button(btn_label, key=sess["id"], use_container_width=True):
            st.session_state.current_session_id = sess["id"]
            st.session_state.messages = get_messages_by_session(sess["id"])
            st.rerun()

    st.divider()
    st.markdown("### 🛠️ Quản lý tài liệu")
    if st.button("🔄 Xóa tài liệu (Upload lại)", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.uploaded_filenames = []
        st.session_state.selected_docs = []
        st.rerun()

    # NEW: Document filtering multiselect — chỉ hiển thị khi đã có file
    if st.session_state.uploaded_filenames:
        st.divider()
        st.markdown("### 🗂️ Lọc theo tài liệu")
        selected = st.multiselect(
            "Chọn file để tìm kiếm\n(bỏ trống = tất cả):",
            options=st.session_state.uploaded_filenames,
            default=st.session_state.selected_docs,
        )
        st.session_state.selected_docs = selected


# ─────────────────────────────────────────────
#  4. Render Header
# ─────────────────────────────────────────────
render_header()


# ─────────────────────────────────────────────
#  5. Khu vực Upload (multi-file)
# ─────────────────────────────────────────────
st.subheader("📁 Upload tài liệu")
uploaded_files = st.file_uploader(
    "Tải lên một hoặc nhiều file (PDF, DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Đang xử lý tài liệu (Splitting, Embedding & Indexing)…"):
        embedder = load_embedder()
        all_chunks = []
        filenames: List[str] = []
        errors: List[str] = []

        progress_bar = st.progress(0, text="Đang chuẩn bị…")

        for idx, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            suffix = os.path.splitext(filename)[1]

            # Ghi ra file tạm để các loader có thể đọc từ đĩa
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                # process_document trả về list[Document] đã gắn metadata
                chunks = process_document(temp_path=tmp_path, filename=filename)
                all_chunks.extend(chunks)
                filenames.append(filename)
            except ValueError as e:
                errors.append(f"❌ '{filename}': {e}")
            finally:
                os.unlink(tmp_path)  # Dọn file tạm ngay sau khi dùng

            progress_bar.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Đã xử lý: {filename}",
            )

        progress_bar.empty()

        for err in errors:
            st.error(err)

        if all_chunks:
            # Tạo MỘT FAISS vector store duy nhất từ tất cả chunks
            st.session_state.vector_store = FAISS.from_documents(all_chunks, embedder)
            st.session_state.uploaded_filenames = filenames
            st.session_state.selected_docs = []  # Reset filter

            # Tự tạo phiên chat mới nếu chưa có
            if st.session_state.current_session_id is None:
                title = "Phiên: " + ", ".join(filenames[:2]) + (
                    "…" if len(filenames) > 2 else ""
                )
                st.session_state.current_session_id = create_new_session(title)
                st.session_state.messages = []

            st.success(
                f"✅ Đã tạo {len(all_chunks)} đoạn văn bản từ "
                f"{len(filenames)} tài liệu. Hãy đặt câu hỏi bên dưới!"
            )
            st.rerun()


# ─────────────────────────────────────────────
#  6. Khu vực Hỏi Đáp (Giao diện Chatbot)
# ─────────────────────────────────────────────
if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("💬 Trò chuyện với tài liệu")

    # Hiển thị badge các file đã upload
    if st.session_state.uploaded_filenames:
        badge_html = " ".join(
            f'<span style="background:#1e3a5f;color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-size:0.78rem;margin-right:4px;">📄 {f}</span>'
            for f in st.session_state.uploaded_filenames
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        filter_label = (
            ", ".join(st.session_state.selected_docs)
            if st.session_state.selected_docs
            else "tất cả tài liệu"
        )
        st.caption(f"🔍 Đang tìm kiếm trong: **{filter_label}**")

    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = create_new_session("Phiên chat mới")

    # 6.1 Hiển thị lại toàn bộ lịch sử chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="display:flex;justify-content:flex-end;margin-bottom:1rem;">
                    <div style="background-color:#007BFF;color:white;padding:10px 16px;
                                border-radius:20px 20px 4px 20px;max-width:75%;
                                line-height:1.5;box-shadow:0 1px 2px rgba(0,0,0,0.1);">
                        {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Ô nhập liệu dạng chat
    user_question = st.chat_input("Nhập câu hỏi của bạn về nội dung tài liệu…")

    if user_question:
        # 6.2 Lưu câu hỏi của User vào STATE và DB
        st.session_state.messages.append({"role": "user", "content": user_question})
        add_message(st.session_state.current_session_id, "user", user_question)

        st.markdown(
            f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:1rem;">
                <div style="background-color:#007BFF;color:white;padding:10px 16px;
                            border-radius:20px 20px 4px 20px;max-width:75%;
                            line-height:1.5;box-shadow:0 1px 2px rgba(0,0,0,0.1);">
                    {user_question}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 6.3 Xây dựng chuỗi lịch sử hội thoại cho prompt
        recent_messages = (
            st.session_state.messages[-5:-1]
            if len(st.session_state.messages) > 1
            else []
        )
        chat_history_str = ""
        for msg in recent_messages:
            role_name = "Người dùng" if msg["role"] == "user" else "AI"
            chat_history_str += f"{role_name}: {msg['content']}\n"

        # 6.4 Xử lý và hiển thị câu trả lời của AI
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            reranked_docs = []
            unique_sources = []

            try:
                # NEW: unpack 3 biến từ run_rag_chain đã cập nhật
                response_stream, reranked_docs, unique_sources = run_rag_chain(
                    user_question,
                    st.session_state.vector_store,
                    chat_history_str,
                )

                # Stream từng token ra màn hình
                for token in response_stream:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                # NEW: Gắn danh sách nguồn vào cuối câu trả lời
                if unique_sources:
                    source_text = ", ".join(unique_sources)
                    full_response += f"\n\n**Nguồn:** [{source_text}]"

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Đã xảy ra lỗi khi xử lý câu hỏi: {e}"
                response_placeholder.error(full_response)

            # NEW: Expander chi tiết trích dẫn từ reranked_docs
            if reranked_docs:
                with st.expander("📚 Xem nguồn trích dẫn (đã rerank)"):
                    for i, doc in enumerate(reranked_docs):
                        page_num = doc.metadata.get("page", 0) + 1
                        file_name = os.path.basename(
                            doc.metadata.get("source", "Tài liệu")
                        )
                        rerank_score = doc.metadata.get("rerank_score", None)

                        score_badge = (
                            f" · `score: {rerank_score:.4f}`"
                            if rerank_score is not None
                            else ""
                        )

                        st.markdown(
                            f"**Nguồn {i + 1} — `{file_name}` · Trang {page_num}**"
                            f"{score_badge}"
                        )
                        st.caption(f"_{doc.page_content[:200]}…_")
                        if i < len(reranked_docs) - 1:
                            st.markdown("---")

        # 6.5 Lưu câu trả lời của AI vào STATE và DB
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        add_message(st.session_state.current_session_id, "assistant", full_response)

        st.rerun()

else:
    # Chưa upload tài liệu — hướng dẫn nhanh
    st.info("👆 Hãy upload ít nhất một tài liệu PDF hoặc DOCX để bắt đầu trò chuyện.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "### 📂 Multi-Document\nUpload nhiều file cùng lúc và hỏi đáp trên toàn bộ."
        )
    with col2:
        st.markdown(
            "### 🔍 Metadata Filtering\nLọc kết quả theo từng file cụ thể trong sidebar."
        )
    with col3:
        st.markdown(
            "### 🏆 Cross-Encoder Reranking\nKết quả được sắp xếp lại bằng AI để chính xác hơn."
        )