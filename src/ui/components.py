import streamlit as st

def render_sidebar():
    """Hiển thị thanh Sidebar bên trái [cite: 389]"""
    with st.sidebar:
        st.title("⚙️ Cài đặt & Thông tin")
        st.info("Mô hình: Qwen2.5:7b\nEmbedding: Multilingual MPNet")
        
        st.markdown("### Hướng dẫn sử dụng:")
        st.markdown("""
        1. Upload file PDF của bạn.
        2. Chờ hệ thống xử lý nội dung.
        3. Đặt câu hỏi để tìm kiếm thông tin.
        """)

def render_header():
    """Hiển thị tiêu đề chính [cite: 394]"""
    st.title("🤖 SmartDoc AI - Trợ lý tài liệu thông minh") 

def render_file_uploader():
    """Hiển thị khu vực upload file và trả về file được chọn"""
    # Thêm 'docx' vào danh sách type
    return st.file_uploader("Tải lên tài liệu của bạn (PDF, DOCX)", type=["pdf", "docx"])

def render_question_input():
    """Hiển thị ô nhập câu hỏi và trả về text người dùng nhập"""
    st.divider()
    st.subheader("💬 Trò chuyện với tài liệu")
    return st.text_input("Nhập câu hỏi của bạn về nội dung PDF:")

def render_response(response: str):
    """Hiển thị kết quả trả về từ AI [cite: 397, 419]"""
    st.markdown("### 💡 Trả lời:")
    st.info(response)