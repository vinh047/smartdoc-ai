import streamlit as st
import os
from src.core.database import (
    get_all_sessions,
    create_new_session,
    delete_session,
    rename_session,
    clear_all_sessions,
    get_uploaded_files,
    get_uploaded_files_info,
    delete_document,
    clear_all_documents,
)
from src.core.document_processor import process_document, load_stores
from src.config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def render_header():
    st.markdown("## SmartDoc System")
    st.caption("Enterprise RAG Architecture - OSSD 2026")


def render_sidebar():
    with st.sidebar:
        # Nếu chưa có session nào, tự tạo 1 cái
        if not st.session_state.get("current_session_id"):
            st.session_state.current_session_id = create_new_session(
                "Phiên làm việc mặc định"
            )

        # Sử dụng các tab có tên chuyên nghiệp, bỏ emoji
        tab1, tab2, tab3 = st.tabs(["Hội thoại", "Dữ liệu", "Hệ thống"])

        # ==========================================
        # TAB 1: QUẢN LÝ LỊCH SỬ CHAT
        # ==========================================
        with tab1:
            if st.button(
                "Tạo cuộc hội thoại mới", use_container_width=True, type="primary"
            ):
                st.session_state.current_session_id = create_new_session(
                    "Phiên làm việc mới"
                )
                st.session_state.messages = []
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            sessions = get_all_sessions()

            if not sessions:
                st.caption("Chưa có dữ liệu hội thoại.")
            else:
                for s in sessions:
                    col1, col2, col3 = st.columns([6, 2, 2])
                    # Nút chọn session
                    if col1.button(
                        f"{s['title'][:15]}...",
                        key=f"btn_{s['id']}",
                        help=str(s["created_at"]),
                        use_container_width=True,
                    ):
                        st.session_state.current_session_id = s["id"]
                        st.rerun()

                    # Nút đổi tên
                    with col2.popover("s"):
                        new_name = st.text_input(
                            "Nhập tên mới", value=s["title"], key=f"rn_{s['id']}"
                        )
                        if st.button(
                            "Lưu thay đổi", key=f"sv_{s['id']}", type="primary"
                        ):
                            rename_session(s["id"], new_name)
                            st.rerun()

                    # Nút xóa
                    if col3.button("X", key=f"del_{s['id']}", type="secondary"):
                        delete_session(s["id"])
                        if st.session_state.current_session_id == s["id"]:
                            st.session_state.current_session_id = None
                        st.rerun()

            st.markdown("---")
            if st.button("Xóa toàn bộ lịch sử", use_container_width=True):
                clear_all_sessions()
                st.session_state.current_session_id = None
                st.rerun()

        # ==========================================
        # TAB 2: QUẢN LÝ TÀI LIỆU
        # ==========================================
        with tab2:
            st.markdown("#### Bộ lọc ngữ cảnh")
            uploaded_files = get_uploaded_files()

            # Xóa file khỏi selected_files nếu file đó không còn tồn tại trong DB
            if "selected_files" in st.session_state:
                st.session_state.selected_files = [
                    f for f in st.session_state.selected_files if f in uploaded_files
                ]

            st.session_state.selected_files = st.multiselect(
                "Chỉ định tài liệu truy vấn (để trống để tìm trên toàn hệ thống):",
                options=list(set(uploaded_files)),
                default=st.session_state.get("selected_files", []),
            )

            st.markdown("---")
            st.markdown("#### Nhập tài liệu mới")
            uploaded_file = st.file_uploader(
                "Định dạng hỗ trợ: PDF, DOCX",
                type=["pdf", "docx"],
                label_visibility="collapsed",
            )

            if uploaded_file and uploaded_file.name not in uploaded_files:
                with st.spinner("Đang xử lý và trích xuất dữ liệu..."):
                    os.makedirs("data", exist_ok=True)
                    temp_path = f"data/temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    c_size = st.session_state.get("ui_chunk_size", DEFAULT_CHUNK_SIZE)
                    c_overlap = st.session_state.get(
                        "ui_chunk_overlap", DEFAULT_CHUNK_OVERLAP
                    )

                    if process_document(
                        temp_path, uploaded_file.name, c_size, c_overlap
                    ):
                        (
                            st.session_state.vector_store,
                            st.session_state.bm25_retriever,
                        ) = load_stores()
                        st.success(f"Đã xử lý: {uploaded_file.name}")
                        st.rerun()

            st.markdown("---")
            st.markdown("#### Quản lý tài liệu đã tải lên")
            files_info = get_uploaded_files_info()

            if not files_info:
                st.caption("Chưa có tài liệu nào trong hệ thống.")
            else:
                for f in files_info:
                    with st.container(border=True):
                        st.markdown(f"**{f['file_name']}**")
                        st.caption(f"Tải lên: {f['uploaded_at']}")
                        if st.button(
                            "Xóa tệp này",
                            key=f"del_doc_{f['id']}",
                            use_container_width=True,
                        ):
                            delete_document(f["file_name"])
                            st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Xóa tất cả tài liệu", use_container_width=True):
                    clear_all_documents()
                    st.rerun()

        # ==========================================
        # TAB 3: CẤU HÌNH HỆ THỐNG
        # ==========================================
        with tab3:
            st.markdown("#### Chiến lược phân mảnh (Chunking)")
            st.session_state.ui_chunk_size = st.slider(
                "Kích thước phân mảnh (Chunk Size)", 500, 2500, DEFAULT_CHUNK_SIZE, 100
            )
            st.session_state.ui_chunk_overlap = st.slider(
                "Độ lấp đầy (Chunk Overlap)", 50, 500, DEFAULT_CHUNK_OVERLAP, 50
            )
            st.caption(
                "Lưu ý: Các thiết lập này chỉ áp dụng cho tài liệu tải lên sau khi thay đổi."
            )

            st.markdown("---")
            st.markdown("#### Trạng thái Module")
            st.info(
                "• Multi-Document RAG: Đang hoạt động\n\n"
                "• Map-Reduce Summarize: Đang hoạt động\n\n"
                "• Hybrid Search (FAISS+BM25): Đang hoạt động\n\n"
                "• Cross-Encoder Reranking: Đang hoạt động"
            )
