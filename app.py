import streamlit as st
import os
import warnings

from src.core.document_processor import load_stores
from src.core.database import init_db, get_messages_by_session, add_message
from src.ui.components import render_sidebar, render_header
from src.engines.main_dispatcher import execute_query

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH TRANG & CSS GIAO DIỆN CHUYÊN NGHIỆP ---
# Bỏ icon để tạo sự tinh gọn
st.set_page_config(page_title="SmartDoc System", layout="wide")

# CSS Tiêm vào để tùy biến UI sang phong cách Enterprise
# Fix lỗi align: Dùng cú pháp chuẩn xác kết hợp với class user-msg
st.markdown(
    """
<style>
    /* Ẩn main menu và footer mặc định của Streamlit cho chuyên nghiệp */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Thiết kế cho tin nhắn của User (Bên phải, màu nền xám nhạt chuyên nghiệp) */
    div[data-testid="stChatMessage"]:has(span.user-msg) {
        flex-direction: row-reverse;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-left: auto;
        margin-right: 0;
        width: fit-content;
        max-width: 85%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    div[data-testid="stChatMessage"]:has(span.user-msg) div[data-testid="stMarkdownContainer"] {
        text-align: right;
        color: #212529;
    }

    /* Ẩn avatar người dùng để tạo cảm giác giống MS Teams / Slack hiện đại */
    div[data-testid="stChatMessage"]:has(span.user-msg) div[data-testid="chatAvatarIcon-user"] {
        display: none;
    }
    div[data-testid="stChatMessage"]:has(span.user-msg) div.st-emotion-cache-1ebukzx {
        display: none;
    }

    /* Thiết kế cho tin nhắn của AI Assistant (Bên trái, nền trắng) */
    div[data-testid="stChatMessage"]:not(:has(span.user-msg)) {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-right: auto;
        margin-left: 0;
        max-width: 90%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    /* Làm gọn lại các nút expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        color: #495057;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- 2. KHỞI TẠO STATE ---
@st.cache_resource(show_spinner="Đang khởi tạo hệ thống truy xuất...")
def setup_system():
    init_db()
    return load_stores()


if "system_initialized" not in st.session_state:
    st.session_state.vector_store, st.session_state.bm25_retriever = setup_system()
    st.session_state.system_initialized = True

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []

# --- 3. RENDER GIAO DIỆN ---
render_header()
render_sidebar()

# KHUNG CHAT CHÍNH
if st.session_state.current_session_id:
    st.session_state.messages = get_messages_by_session(
        st.session_state.current_session_id
    )

# Hiển thị tin nhắn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            # Gắn cờ <span> để CSS ép lề phải
            st.markdown(
                f'<span class="user-msg"></span>{msg["content"]}',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])

# Input Chat
user_question = st.chat_input(
    "Nhập câu hỏi truy vấn dữ liệu (vd: Tóm tắt chương 1, So sánh...)"
)

if user_question:
    # 1. Hiện câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": user_question})
    add_message(st.session_state.current_session_id, "user", user_question)

    with st.chat_message("user"):
        st.markdown(
            f'<span class="user-msg"></span>{user_question}', unsafe_allow_html=True
        )

    # 2. Xử lý câu trả lời
    with st.chat_message("assistant"):
        history_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]]
        )

        response_stream, source_docs, meta_info = execute_query(
            query=user_question,
            history_str=history_str,
            vector_store=st.session_state.vector_store,
            bm25_retriever=st.session_state.bm25_retriever,
            session_id=st.session_state.current_session_id,
            selected_files=st.session_state.selected_files,
        )

        full_response = st.write_stream(response_stream)

        # Hiển thị Citation (Trích dẫn) chuyên nghiệp
        if source_docs:
            with st.expander("Nguồn trích dẫn & Thông tin tham chiếu"):
                for i, doc in enumerate(source_docs):
                    file_name = doc.metadata.get("source", "Không rõ nguồn")
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**[{i + 1}] Tài liệu: `{file_name}` (Trang {page})**")
                    st.caption(f"Trích đoạn: {doc.page_content}")
                    st.markdown("---")

    # 3. Lưu DB
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "meta_data": meta_info}
    )
    add_message(
        st.session_state.current_session_id, "assistant", full_response, meta_info
    )
