import pytest
from unittest.mock import patch
from src.core.vector_manager import build_vector_store


def test_build_vector_store_empty_chunks():
    """Kiểm tra: Nếu tài liệu trắng tinh (không có chữ), phải báo lỗi ValueError."""
    with pytest.raises(ValueError, match="Không trích xuất được nội dung"):
        build_vector_store([])


# Dùng patch chặn HuggingFace và FAISS gọi ra Internet/GPU
@patch("src.core.vector_manager.FAISS.from_documents")
@patch("src.core.vector_manager.HuggingFaceEmbeddings")
def test_build_vector_store_success(mock_hf, mock_faiss):
    """Kiểm tra: Build vector thành công từ dữ liệu hợp lệ."""

    # Giả lập cái kho FAISS trả về thành công
    mock_faiss.return_value = "FAISS_STORE_XIN_SO"

    # Dữ liệu đầu vào giả
    fake_chunks = [
        {"page_content": "Vinamilk lãi lớn", "metadata": {"source": "doc1.pdf"}},
        {"page_content": "Chi phí giảm", "metadata": {"source": "doc1.pdf"}},
    ]

    # Chạy hàm
    result = build_vector_store(fake_chunks)

    # Khẳng định (Assert)
    assert result == "FAISS_STORE_XIN_SO"
    mock_hf.assert_called_once()  # Đảm bảo đã khởi tạo Embedder
    mock_faiss.assert_called_once()  # Đảm bảo đã gọi FAISS build
