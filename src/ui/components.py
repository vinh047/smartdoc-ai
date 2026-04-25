import json
import streamlit as st
from typing import Dict, Any


def inject_custom_css():
    """Bơm CSS tùy chỉnh thích nghi với cả Light và Dark mode."""
    st.markdown(
        """
        <style>
        section[data-testid="stMain"] [data-testid="stExpander"]:first-of-type {
            position: sticky;
            top: 3.5rem; 
            z-index: 999;
        }
        section[data-testid="stMain"] [data-testid="stExpander"]:first-of-type details {
            /* Dùng màu nền mờ để thích nghi mọi giao diện */
            background-color: rgba(128, 128, 128, 0.1); 
            padding: 5px;
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Hiển thị thông tin dự án trên thanh Sidebar."""
    st.title("⚙️ Cài đặt & Thông tin")
    st.info("Mô hình: Qwen2.5:7b\nEmbedding: Multilingual MPNet")

    st.markdown("### Hướng dẫn sử dụng:")
    st.markdown(
        """
    1. Upload file PDF/DOCX của bạn.
    2. Chờ hệ thống xử lý nội dung.
    3. Đặt câu hỏi để tìm kiếm thông tin.
    """
    )


def render_header():
    """Hiển thị tiêu đề chính của ứng dụng."""
    st.title("🤖 SmartDoc AI - Trợ lý tài liệu thông minh")


def render_chat_message(msg: Dict[str, Any]):
    """
    Giải mã và hiển thị một tin nhắn trong lịch sử chat.
    Tự động nhận diện tin nhắn thường hoặc tin nhắn JSON chia 2 cột (Split-Screen).
    """
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return

        # Xử lý tin nhắn của AI (Giải mã JSON để vẽ 2 cột)
        try:
            data = json.loads(msg["content"])
            if isinstance(data, dict) and data.get("is_split"):
                col1, col2 = st.columns(2)

                # Cột 1: Standard RAG
                with col1:
                    st.markdown("### 🟢 Standard RAG")
                    st.info(data.get("rag_answer", ""))
                    if data.get("rag_citations"):
                        # Dùng popover (nút bấm hiện cửa sổ) để click xem context gốc
                        with st.popover("📚 Click xem Context gốc"):
                            for i, cite in enumerate(data["rag_citations"]):
                                st.markdown(
                                    f"**📍 Nguồn {i+1} | Trang {cite.get('page', 0)}**"
                                )
                                # Vẽ khung Highlight cho đoạn văn
                                highlighted_text = f"""
                                <div style="background-color: rgba(255, 212, 59, 0.15); border-left: 4px solid #ffd43b; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 0.9em; line-height: 1.5;">
                                    {cite.get('content', '')}
                                </div>
                                """
                                st.markdown(highlighted_text, unsafe_allow_html=True)

                # Cột 2: Advanced CoRAG
                with col2:
                    st.markdown("### 🚀 Advanced CoRAG")
                    st.warning(data.get("corag_answer", ""))
                    st.metric(
                        "Độ tự tin", f"{data.get('corag_confidence', 0)*100:.0f}%"
                    )

                    if data.get("citations"):  # Lấy mảng citations của CoRAG
                        with st.popover("🎯 Nguồn đã tinh chỉnh (Cross-Encoder)"):
                            for i, cite in enumerate(data["citations"]):
                                score_badge = f"<span style='background:#ff4b4b; color:white; padding:2px 6px; border-radius:10px; font-size:0.8em;'>Điểm: {cite.get('rerank_score', 0):.2f}</span>"
                                st.markdown(
                                    f"**📍 Nguồn {i+1} | Trang {cite.get('page', 'N/A')}** {score_badge}",
                                    unsafe_allow_html=True,
                                )

                                highlighted_text = f"""
                                <div style="background-color: rgba(255, 75, 75, 0.1); border-left: 4px solid #ff4b4b; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 0.9em; line-height: 1.5;">
                                    {cite.get('content', '')}
                                </div>
                                """
                                st.markdown(highlighted_text, unsafe_allow_html=True)

                    if data.get("corag_suggestions"):
                        st.markdown(
                            "**💡 Gợi ý:** " + ", ".join(data["corag_suggestions"])
                        )
            else:
                st.markdown(msg["content"])  # Fallback nếu JSON lạ
        except json.JSONDecodeError:
            st.markdown(msg["content"])  # Fallback nếu là Text bình thường
