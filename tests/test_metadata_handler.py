import pytest
from src.core.metadata_handler import MetadataManager
from langchain_core.documents import Document


def test_create_metadata():
    """Kiểm tra bộ sinh Metadata có tạo đủ các trường cần thiết không."""
    manager = MetadataManager()
    result = manager.create_metadata("baocao.pdf", "report")

    assert result["source"] == "baocao.pdf"
    assert result["doc_type"] == "report"
    assert "upload_date" in result
    assert "doc_id" in result  # Cực kỳ quan trọng để Database nhận diện


def test_attach_metadata_to_docs():
    """Kiểm tra việc gắn metadata không được làm mất số trang (page number) gốc."""
    manager = MetadataManager()

    # Giả lập 2 chunk tài liệu vừa được Langchain cắt ra (đã có sẵn số trang)
    docs = [
        Document(page_content="Nội dung trang 1", metadata={"page": 1}),
        Document(page_content="Nội dung trang 2", metadata={"page": 2}),
    ]

    new_meta = manager.create_metadata("tailieu_chung.pdf")
    updated_docs = manager.attach_metadata_to_docs(docs, new_meta)

    # Đảm bảo metadata cũ (page) KHÔNG bị ghi đè mất
    assert updated_docs[0].metadata["page"] == 1
    assert updated_docs[1].metadata["page"] == 2

    # Đảm bảo metadata mới đã được đắp vào thành công
    assert updated_docs[0].metadata["source"] == "tailieu_chung.pdf"
    assert "doc_id" in updated_docs[1].metadata
