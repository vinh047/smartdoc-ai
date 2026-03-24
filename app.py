import streamlit as st
import os

# Import logic xử lý lõi
from src.core import process_document
from src.core import run_rag_chain

# Import các hàm tương tác Database
from src.core.database import (
    init_db, 
    create_new_session, 
    get_all_sessions,
    add_message, 
    get_messages_by_session
)

# Import các component UI
from src.ui import (
    render_sidebar,
    render_header,
    render_file_uploader
)

# 1. Khởi tạo Database (Chỉ chạy 1 lần để tạo file .db nếu chưa có)
init_db()

@st.cache_resource(show_spinner=False)
def load_and_cache_document(file_path):
    """Hàm này sẽ lưu kết quả xử lý PDF vào RAM, không chạy lại ở các lần click sau"""
    return process_document(file_path)

# Cấu hình cơ bản
st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

# 2. Khởi tạo các State cần thiết
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# 3. Khu vực Sidebar (Quản lý Phiên trò chuyện & Tài liệu)
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
    # Lấy danh sách sessions từ DB và render thành các nút bấm
    sessions = get_all_sessions()
    for sess in sessions:
        btn_label = f"📝 {sess['title']} ({sess['created_at'][:10]})"
        if st.button(btn_label, key=sess['id'], use_container_width=True):
            st.session_state.current_session_id = sess['id']
            st.session_state.messages = get_messages_by_session(sess['id'])
            st.rerun()
            
    st.divider()
    st.markdown("### 🛠️ Quản lý tài liệu")
    if st.button("🔄 Xóa tài liệu (Upload lại)", use_container_width=True):
        st.session_state.vector_store = None
        st.rerun()

# 4. Render Header
render_header()

# 5. Khu vực Upload
uploaded_file = render_file_uploader()

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Đang xử lý tài liệu (Splitting & Embedding)..."):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Gọi logic xử lý file
        st.session_state.vector_store = load_and_cache_document(temp_path)
        
        os.remove(temp_path)
        st.success("Tài liệu đã được xử lý thành công! Hãy đặt câu hỏi bên dưới.")

# 6. Khu vực Hỏi Đáp (Giao diện Chatbot)
if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("💬 Trò chuyện với tài liệu")
    
    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = create_new_session("Phiên chat mới")
    
    # 6.1 Hiển thị lại toàn bộ lịch sử chat ở màn hình chính
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                    <div style="background-color: #007BFF; color: white; padding: 10px 16px; border-radius: 20px 20px 4px 20px; max-width: 75%; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        {msg['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
            
    # Ô nhập liệu dạng chat
    user_question = st.chat_input("Nhập câu hỏi của bạn về nội dung tài liệu...")
    
    if user_question:
        # 6.2 Lưu câu hỏi của User vào STATE và DB
        st.session_state.messages.append({"role": "user", "content": user_question})
        add_message(st.session_state.current_session_id, "user", user_question)
        
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                <div style="background-color: #007BFF; color: white; padding: 10px 16px; border-radius: 20px 20px 4px 20px; max-width: 75%; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {user_question}
                </div>
            </div>
        """, unsafe_allow_html=True)
            
        # 6.3 Xử lý và hiển thị câu trả lời của AI
        with st.chat_message("assistant"):
            recent_messages = st.session_state.messages[-5:-1] if len(st.session_state.messages) > 1 else []
            chat_history_str = ""
            for msg in recent_messages:
                role_name = "Người dùng" if msg["role"] == "user" else "AI"
                chat_history_str += f"{role_name}: {msg['content']}\n"
                
            # Hứng 2 biến: Luồng chữ VÀ danh sách nguồn
            response_stream, source_docs = run_rag_chain(user_question, st.session_state.vector_store, chat_history_str)
            
            # AI nhả chữ ra màn hình
            full_response = st.write_stream(response_stream)
            
            # HIỂN THỊ NGUỒN (CITATIONS)
            if source_docs:
                with st.expander("📚 Xem nguồn trích dẫn"):
                    for i, doc in enumerate(source_docs):
                        # PDFPlumber tự động lấy số trang (bắt đầu từ 0)
                        page_num = doc.metadata.get('page', 0) + 1 
                        file_name = os.path.basename(doc.metadata.get('source', 'Tài liệu'))
                        
                        st.markdown(f"**Nguồn {i+1} (Trang {page_num} - `{file_name}`)**")
                        # Hiển thị 150 ký tự đầu tiên để người dùng kiểm chứng
                        st.caption(f"_{doc.page_content[:150]}..._")
                
        # 6.4 Lưu câu trả lời của AI vào STATE và DB
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        add_message(st.session_state.current_session_id, "assistant", full_response)