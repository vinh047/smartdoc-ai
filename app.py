"""
SmartDoc AI — Main Streamlit Application
=========================================
Advanced RAG pipeline với: Multi-document upload, Metadata filtering, Cross-Encoder reranking, Persistent chat history.
"""

import os
import json
import streamlit as st
from langchain_core.documents import Document as LangchainDocument
from langchain_ollama import OllamaLLM

# Core Modules
from src.core.document_processor import process_document_data
from src.core.vector_manager import build_vector_store
from src.core.rag_engine import run_rag_chain
from src.core.engines.advanced_rag import run_advanced_rag_pipeline
from src.config import LLM_CONFIG
from src.core.metadata_handler import MetadataManager
from src.core.database import (
    init_db,
    create_new_session,
    get_all_sessions,
    add_message,
    get_messages_by_session,
    clear_vector_store,
    delete_session_permanently,
    update_session_title,
)

# UI Components
from src.ui.components import (
    render_sidebar,
    render_header,
    render_chat_message,
    inject_custom_css,
)

# =====================================================================
# 1. KHỞI TẠO HỆ THỐNG & TRẠNG THÁI (STATE)
# =====================================================================
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")
init_db()


def init_app_state():
    """Khởi tạo toàn bộ biến bộ nhớ của Streamlit."""
    states = {
        "vector_store": None,
        "documents": None,
        "messages": [],
        "current_session_id": None,
        "sessions_data": {},
        "clicked_suggestion": None,
        "processed_files": set(),
    }
    for key, val in states.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_app_state()
inject_custom_css()


def set_suggestion(q):
    """Callback: Hứng câu hỏi khi người dùng bấm nút gợi ý."""
    st.session_state.clicked_suggestion = q


@st.dialog("⚠️ Xác nhận xóa phiên chat")
def confirm_delete_session(session_id):
    st.warning(
        "Bạn có chắc chắn muốn xóa vĩnh viễn cuộc trò chuyện này không? Hành động này không thể hoàn tác."
    )
    col1, col2 = st.columns(2)
    if col1.button("✔️ Xác nhận Xóa", type="primary", use_container_width=True):
        try:  # BẮT ĐẦU BỌC LỖI
            delete_session_permanently(session_id)
            if session_id in st.session_state.sessions_data:
                del st.session_state.sessions_data[session_id]
            if st.session_state.current_session_id == session_id:
                init_app_state()
            st.toast("Đã xóa phiên chat thành công!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi xóa: {e}")
    if col2.button("❌ Hủy bỏ", use_container_width=True):
        st.rerun()


@st.dialog("⚠️ Xác nhận làm mới hệ thống")
def confirm_clear_vector_store():
    st.error(
        "Hành động này sẽ xóa TOÀN BỘ tài liệu đã tải lên trong hệ thống. Bạn có chắc chắn không?"
    )
    col1, col2 = st.columns(2)
    if col1.button("✔️ Chắc chắn Xóa", type="primary", use_container_width=True):
        try:  # BẮT ĐẦU BỌC LỖI
            clear_vector_store()
            st.session_state.vector_store = None
            st.session_state.documents = None
            st.session_state.processed_files = set()
            st.toast("Đã làm mới hệ thống tài liệu!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi làm mới: {e}")
    if col2.button("❌ Hủy bỏ", use_container_width=True):
        st.rerun()


# =====================================================================
# 2. RENDER SIDEBAR (Điều hướng & Cấu hình)
# =====================================================================
with st.sidebar:
    render_sidebar()
    st.divider()

    st.markdown("### ⚙️ Cấu hình cắt chữ (Chunking)")
    c_size = st.slider("Chunk Size", 500, 2000, 1000, 100, key="sz")
    c_overlap = st.slider("Chunk Overlap", 50, 500, 100, 50, key="ov")
    st.divider()

    st.markdown("### 💬 Lịch sử trò chuyện")
    if st.button("➕ Cuộc trò chuyện mới", type="primary", use_container_width=True):
        st.session_state.current_session_id = create_new_session("Phiên chat mới")
        st.session_state.messages = []
        st.rerun()

    for sess in get_all_sessions():
        col_select, col_del = st.sidebar.columns([0.85, 0.15])
        with col_select:
            if st.button(
                f"📝 {sess['title'][:15]}...",
                key=f"sess_{sess['id']}",
                use_container_width=True,
            ):
                st.session_state.current_session_id = sess["id"]
                st.session_state.messages = get_messages_by_session(sess["id"])
                sess_info = st.session_state.sessions_data.get(sess["id"], {})
                st.session_state.vector_store = sess_info.get("vs", None)
                st.session_state.documents = sess_info.get("docs", None)
                st.session_state.processed_files = sess_info.get(
                    "processed_files", set()
                )
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{sess['id']}"):
                confirm_delete_session(sess["id"])

    st.divider()

    if st.button("🔄 Xóa Vector DB (Upload lại)", use_container_width=True):
        confirm_clear_vector_store()

# =====================================================================
# 3. MÀN HÌNH CHÍNH: QUẢN LÝ TÀI LIỆU (UPLOAD)
# =====================================================================
render_header()
is_expanded = st.session_state.vector_store is None
with st.expander("📂 Quản lý Tải lên & Bộ lọc tài liệu", expanded=is_expanded):
    uploaded_files = st.file_uploader(
        "Kéo thả tài liệu vào đây (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    current_file_names = (
        set([f.name for f in uploaded_files]) if uploaded_files else set()
    )

    if uploaded_files and current_file_names != st.session_state.processed_files:
        with st.spinner("Đang xử lý và cập nhật kho tài liệu..."):
            try:
                all_chunks, all_docs = [], []
                meta_manager = MetadataManager()

                for file in uploaded_files:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    chunks, _ = process_document_data(temp_path, c_size, c_overlap)
                    base_meta = meta_manager.create_metadata(file.name)
                    for c in chunks:
                        c["metadata"].update(base_meta)

                    docs = [
                        LangchainDocument(
                            page_content=c["page_content"], metadata=c["metadata"]
                        )
                        for c in chunks
                    ]
                    all_chunks.extend(chunks)
                    all_docs.extend(docs)
                    os.remove(temp_path)

                st.session_state.vector_store = build_vector_store(all_chunks)
                st.session_state.documents = all_docs
                st.session_state.processed_files = current_file_names

                if st.session_state.current_session_id is None:
                    st.session_state.current_session_id = create_new_session(
                        "Phiên chat mới"
                    )

                st.session_state.sessions_data[st.session_state.current_session_id] = {
                    "vs": st.session_state.vector_store,
                    "docs": st.session_state.documents,
                    "processed_files": st.session_state.processed_files,
                }
                st.success("Đã cập nhật hệ thống tài liệu!")
                st.rerun()
            except Exception as e:
                # HIỂN THỊ LỖI UX/UI
                st.error(f"🚨 Đã xảy ra lỗi khi đọc tài liệu: {str(e)}")
                st.info(
                    "💡 Gợi ý: Kiểm tra xem file PDF có bị hỏng hoặc bị khóa mật khẩu không."
                )

    search_filter = None
    if st.session_state.vector_store is not None:
        st.divider()
        available_sources = list(
            set(
                [
                    doc.metadata.get("source")
                    for doc in st.session_state.documents
                    if doc.metadata.get("source")
                ]
            )
        )
        selected_source = st.selectbox(
            "🎯 Lọc tài liệu tìm kiếm:", ["Tất cả tài liệu"] + available_sources
        )
        search_filter = (
            [selected_source] if selected_source != "Tất cả tài liệu" else None
        )

# =====================================================================
# 4. MÀN HÌNH CHÍNH: KHU VỰC CHAT & SPLIT-SCREEN RAG
# =====================================================================
show_chat_area = (st.session_state.current_session_id is not None) or (
    st.session_state.vector_store is not None
)

if show_chat_area:
    st.subheader("💬 Trò chuyện với tài liệu (Split-Screen Comparison)")

    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = create_new_session("Phiên chat mới")

    # 4.1. Luôn luôn hiển thị lịch sử trò chuyện (Dù RAM đã bị xóa file)
    for msg in st.session_state.messages:
        render_chat_message(msg)

    # 4.2. Hứng luồng câu hỏi mới (Chỉ hiện ô nhập nếu có Vector Store)
    user_question = None
    if st.session_state.vector_store is not None:
        user_question = st.chat_input("Nhập câu hỏi của bạn về nội dung tài liệu...")
        if st.session_state.clicked_suggestion:
            user_question = st.session_state.clicked_suggestion
            st.session_state.clicked_suggestion = None

    # 4.3. Phân luồng UI: Báo thiếu file HOẶC chạy AI
    if st.session_state.vector_store is None:
        if len(st.session_state.messages) > 0:
            st.warning(
                "⚠️ Phiên trò chuyện này thuộc về lần truy cập trước. Để tối ưu bộ nhớ hệ thống, tài liệu tạm đã được giải phóng. Vui lòng **tải lên lại tài liệu cũ** ở khung phía trên để tiếp tục hỏi đáp."
            )
        else:
            st.info(
                "👈 Vui lòng tải lên tài liệu ở khung phía trên để bắt đầu trò chuyện."
            )

    elif user_question:

        # [TÍNH NĂNG MỚI] Tự động đổi tên phiên chat nếu đây là câu hỏi đầu tiên
        if len(st.session_state.messages) == 0:
            short_title = (
                user_question[:25] + "..." if len(user_question) > 25 else user_question
            )
            update_session_title(st.session_state.current_session_id, short_title)

        st.session_state.messages.append({"role": "user", "content": user_question})
        add_message(st.session_state.current_session_id, "user", user_question)

        with st.chat_message("user"):
            st.markdown(user_question)

        recent_msgs = (
            st.session_state.messages[-4:-1]
            if len(st.session_state.messages) > 1
            else []
        )
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

        col1, col2 = st.columns(2)
        std_answer = "Lỗi sinh câu trả lời"
        source_docs = []
        adv_data = {}

        # ---> CỘT 1: Standard RAG (TV3)
        with col1:
            st.markdown("### 🟢 Standard RAG (TV3)")
            with st.chat_message("assistant"):
                with st.spinner("Đang truy xuất (Hybrid)..."):
                    try:
                        res_stream, source_docs = run_rag_chain(
                            user_question,
                            st.session_state.vector_store,
                            st.session_state.documents,
                            "",
                            search_filter,
                        )
                        std_answer = st.write_stream(res_stream)

                        if source_docs:
                            with st.popover("📚 Click xem Context gốc"):
                                for i, doc in enumerate(source_docs):
                                    st.markdown(
                                        f"**📍 Nguồn {i+1} | Trang {doc.metadata.get('page', 0)}**"
                                    )
                                    highlighted_text = f"""<div style="background-color: rgba(255, 212, 59, 0.15); border-left: 4px solid #ffd43b; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 0.9em;">{doc.page_content}</div>"""
                                    st.markdown(
                                        highlighted_text, unsafe_allow_html=True
                                    )
                    except Exception as e:
                        st.error(f"🔌 Lỗi kết nối AI (TV3): {e}")
                        std_answer = "Xin lỗi, Standard RAG đang gặp sự cố."

        # ---> CỘT 2: Advanced CoRAG (TV5)
        with col2:
            st.markdown("### 🚀 Advanced CoRAG (TV5)")
            with st.chat_message("assistant"):
                with st.spinner("Đang chạy Cross-Encoder & Self-RAG..."):
                    try:
                        llm_adv = OllamaLLM(model=LLM_CONFIG["model"], temperature=0.0)
                        adv_data = run_advanced_rag_pipeline(
                            user_question,
                            st.session_state.vector_store,
                            st.session_state.documents,
                            llm_adv,
                            history_str,
                            search_filter,
                        )

                        st.markdown(adv_data.get("answer", "Lỗi sinh câu trả lời"))
                        st.divider()
                        st.metric(
                            "Độ tự tin (Self-RAG Score)",
                            f"{adv_data.get('confidence_score', 0)*100:.0f}%",
                        )

                        if adv_data.get("citations"):
                            with st.popover("🎯 Nguồn đã tinh chỉnh (Cross-Encoder)"):
                                for i, cite in enumerate(adv_data.get("citations")):
                                    score_badge = f"<span style='background:#ff4b4b; color:white; padding:2px 6px; border-radius:10px; font-size:0.8em;'>Điểm: {cite.get('rerank_score', 0):.2f}</span>"
                                    st.markdown(
                                        f"**📍 Nguồn {i+1} | Trang {cite.get('page', 'N/A')}** {score_badge}",
                                        unsafe_allow_html=True,
                                    )
                                    highlighted_text = f"""<div style="background-color: rgba(255, 75, 75, 0.1); border-left: 4px solid #ff4b4b; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 0.9em;">{cite.get('content', '')}</div>"""
                                    st.markdown(
                                        highlighted_text, unsafe_allow_html=True
                                    )

                        if adv_data.get("suggested_questions"):
                            st.markdown("**💡 Gợi ý câu hỏi:**")
                            for idx, sq in enumerate(
                                adv_data.get("suggested_questions")
                            ):
                                st.button(
                                    sq,
                                    key=f"btn_sugg_{len(st.session_state.messages)}_{idx}",
                                    on_click=set_suggestion,
                                    args=(sq,),
                                )
                    except ConnectionError:
                        st.error(
                            "🔌 Không thể kết nối với Ollama. Vui lòng bật phần mềm Ollama ở Taskbar."
                        )
                    except Exception as e:
                        st.error(f"⚠️ Lỗi hệ thống Advanced CoRAG: {e}")

        # 4.4. Đóng gói dữ liệu và Lưu Database
        try:
            rag_citations = [
                {"page": d.metadata.get("page", 0), "content": d.page_content}
                for d in (source_docs or [])
            ]
            split_data = {
                "is_split": True,
                "rag_answer": std_answer,
                "rag_citations": rag_citations,
                "corag_answer": adv_data.get("answer", "Lỗi sinh câu trả lời"),
                "corag_confidence": adv_data.get("confidence_score", 0),
                "corag_suggestions": adv_data.get("suggested_questions", []),
                "citations": adv_data.get("citations", []),
            }

            final_answer_json = json.dumps(split_data, ensure_ascii=False)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer_json}
            )
            add_message(
                st.session_state.current_session_id, "assistant", final_answer_json
            )
        except Exception as e:
            st.toast("⚠️ Đã xảy ra lỗi khi lưu lịch sử vào Database", icon="⚠️")
            print(f"Lỗi lưu DB: {e}")
