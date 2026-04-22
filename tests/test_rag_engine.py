import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.core.rag_engine import setup_hybrid_retriever


@patch("src.core.rag_engine.EnsembleRetriever")
@patch("src.core.rag_engine.BM25Retriever")
def test_setup_hybrid_retriever_no_filter(mock_bm25, mock_ensemble):
    """Test 1: Khi chọn 'Tất cả tài liệu', hệ thống phải nạp toàn bộ."""
    mock_vector_store = MagicMock()
    mock_docs = [
        Document(page_content="Text 1", metadata={"source": "A.pdf"}),
        Document(page_content="Text 2", metadata={"source": "B.pdf"}),
    ]

    # Chạy hàm không truyền search_filter
    setup_hybrid_retriever(mock_vector_store, mock_docs, search_filter=None)

    # Đảm bảo BM25 được khởi tạo với toàn bộ tài liệu (cả A và B)
    mock_bm25.from_documents.assert_called_once_with(mock_docs)

    # Đảm bảo FAISS được gọi với thông số k=3 chuẩn
    mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})

    # Đảm bảo Ensemble kết hợp với tỷ lệ 50-50
    mock_ensemble.assert_called_once()
    assert mock_ensemble.call_args[1]["weights"] == [0.5, 0.5]


@patch("src.core.rag_engine.EnsembleRetriever")
@patch("src.core.rag_engine.BM25Retriever")
def test_setup_hybrid_retriever_with_filter(mock_bm25, mock_ensemble):
    """Test 2: Khi người dùng chọn file 'A.pdf' ở Sidebar, hệ thống phải loại bỏ 'B.pdf'."""
    mock_vector_store = MagicMock()
    doc1 = Document(page_content="Text 1", metadata={"source": "A.pdf"})
    doc2 = Document(page_content="Text 2", metadata={"source": "B.pdf"})
    mock_docs = [doc1, doc2]

    # Giả lập người dùng chọn lọc A.pdf
    search_filter = ["A.pdf"]
    setup_hybrid_retriever(mock_vector_store, mock_docs, search_filter=search_filter)

    # BM25 CHỈ được phép khởi tạo với doc1 (Đã loại bỏ B.pdf)
    mock_bm25.from_documents.assert_called_once_with([doc1])

    # Kiểm tra xem FAISS retriever có được gắn bộ lọc (Lambda function) vào không
    call_kwargs = mock_vector_store.as_retriever.call_args[1]["search_kwargs"]
    assert "filter" in call_kwargs
    assert callable(call_kwargs["filter"])  # Chắc chắn là có truyền hàm lọc
