from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm_provider import get_llm
from src.core.database import SessionLocal, Document, DocumentChunk


def run_summarize(query: str, selected_files: list):
    """Engine chuyên biệt dùng Map-Reduce để tóm tắt file mà không bị giới hạn Context Window"""
    if not selected_files:
        yield "⚠️ Bạn cần chọn ít nhất một tài liệu ở cột trái để tôi có thể đọc và tóm tắt toàn bộ nội dung nhé!"
        return

    db = SessionLocal()
    try:
        llm = get_llm()
        yield "🔄 Đang tải toàn bộ dữ liệu từ file... \n\n"

        # 1. Lấy toàn bộ chunks của file được chọn từ SQLite
        chunks = (
            db.query(DocumentChunk)
            .join(Document)
            .filter(Document.file_name.in_(selected_files))
            .all()
        )
        if not chunks:
            yield "❌ Không tìm thấy dữ liệu cho file này."
            return

        full_text = "\n".join([c.content for c in chunks])

        # 2. Thuật toán Reduce Cực Hạn (Vì Local LLM xử lý gộp quá lớn sẽ bị tràn RAM)
        # Thay vì tóm tắt từng chunk (Map) rất chậm, ta cắt văn bản thành các đoạn trung bình,
        # tóm tắt từng đoạn, rồi gộp lại (Reduce). Giới hạn max 15000 ký tự cho mô hình 7B.

        truncated_text = (
            full_text[:15000] + "\n...[Đã cắt bớt để tối ưu]"
            if len(full_text) > 15000
            else full_text
        )

        prompt = PromptTemplate(
            template="""Bạn là một chuyên gia tổng hợp tài liệu xuất sắc.
            Dựa trên yêu cầu của người dùng: "{query}"

            Hãy tóm tắt chi tiết và toàn diện nội dung tài liệu sau.
            BẮT BUỘC sử dụng Markdown (Tiêu đề, in đậm, gạch đầu dòng) để trình bày đẹp mắt.

            TÀI LIỆU CHÍNH:
            {context}

            BÁO CÁO TÓM TẮT:""",
            input_variables=["query", "context"],
        )

        chain = prompt | llm | StrOutputParser()
        for chunk in chain.stream({"query": query, "context": truncated_text}):
            yield chunk

    finally:
        db.close()
