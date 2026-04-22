import pytest
from unittest.mock import patch
from src.core.document_processor import process_document_data


def test_process_document_data_file_not_found():
    """Kiểm tra: Hệ thống phải báo lỗi FileNotFoundError nếu file không tồn tại."""
    with pytest.raises(FileNotFoundError, match="Không tìm thấy file"):
        process_document_data("file_ma.pdf")


@patch("src.core.document_processor.os.path.exists")
def test_process_document_data_invalid_extension(mock_exists):
    """Kiểm tra: Ném file .txt vào thì hệ thống phải chặn lại ngay."""
    # Bơm dữ liệu giả: Giả vờ như file có tồn tại trên ổ cứng
    mock_exists.return_value = True

    with pytest.raises(ValueError, match="Chỉ hỗ trợ file PDF và DOCX"):
        process_document_data("test_file.txt")


@patch("src.core.document_processor.os.path.exists")
@patch("src.core.document_processor.process_pdf")
def test_process_document_data_routes_to_pdf(mock_process_pdf, mock_exists):
    """Kiểm tra: Nếu đuôi file là .pdf, hàm process_pdf phải được gọi."""
    mock_exists.return_value = True

    # Giả lập kết quả trả về của process_pdf
    mock_process_pdf.return_value = (["chunk_1", "chunk_2"], [])

    chunks, images = process_document_data("test.pdf", 1000, 100)

    # Khẳng định (Assert) kết quả
    assert len(chunks) == 2
    assert chunks[0] == "chunk_1"
    mock_process_pdf.assert_called_once()  # Đảm bảo hàm pdf đã được chạy đúng 1 lần
