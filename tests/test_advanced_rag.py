import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from src.core.engines.advanced_rag import (
    decompose_query,
    evaluate_with_self_rag,
    run_advanced_rag_pipeline,
)

# =====================================================================
# 1. TEST CÁC HÀM TIỆN ÍCH NHỎ (UNIT TESTS)
# =====================================================================


def test_decompose_query():
    """Kiểm tra: Hàm viết lại câu hỏi có trích xuất đúng JSON từ LLM không."""

    def mock_llm_invoke(prompt):
        class FakeMsg:
            content = '["Ý 1 là gì?", "Ý 2 là gì?"]'

        return FakeMsg()

    # Bơm LLM giả vào hàm
    result = decompose_query("Câu hỏi gốc", RunnableLambda(mock_llm_invoke))

    assert len(result) == 2
    assert result[0] == "Ý 1 là gì?"
    assert result[1] == "Ý 2 là gì?"


def test_evaluate_with_self_rag():
    """Kiểm tra: Hàm Self-RAG có bóc tách đúng điểm Confidence Score không."""

    def mock_llm_invoke(prompt):
        class FakeMsg:
            content = (
                '{"confidence_score": 0.95, "suggested_questions": ["Hỏi thêm 1?"]}'
            )

        return FakeMsg()

    result = evaluate_with_self_rag(
        "query", "context", "answer", RunnableLambda(mock_llm_invoke)
    )

    assert result["confidence_score"] == 0.95
    assert len(result["suggested_questions"]) == 1


# =====================================================================
# 2. TEST TOÀN BỘ PIPELINE (INTEGRATION TEST)
# =====================================================================


@patch("src.core.engines.advanced_rag.setup_hybrid_retriever")
def test_run_advanced_rag_pipeline(mock_setup_retriever):
    """
    Kiểm thử luồng CoRAG hoàn chỉnh: Rewrite -> Retrieve -> Rerank -> Generate -> Evaluate.
    Áp dụng kỹ thuật Mocking toàn diện để không tốn tài nguyên thật.
    """

    # --- 2.1 MOCK LLM (Cách của cậu viết siêu hay, tôi giữ nguyên logic) ---
    def mock_llm_invoke(prompt_val):
        prompt_str = str(prompt_val)

        class FakeAIMessage:
            def __init__(self, content):
                self.content = content

        # Bắt Prompt chia nhỏ câu hỏi
        if "CHỈ TRẢ VỀ MẢNG JSON" in prompt_str:
            return FakeAIMessage('["SmartDoc dùng thuật toán gì?", "CoRAG là gì?"]')

        # Bắt Prompt đánh giá Self-RAG
        if "confidence_score" in prompt_str:
            return FakeAIMessage(
                '{"confidence_score": 0.96, "suggested_questions": ["Tại sao cần Re-ranking?"]}'
            )

        # Bắt Prompt trả lời chính (Generation)
        return FakeAIMessage(
            "SmartDoc dùng Cross-Encoder để Re-ranking. CoRAG có thêm Sliding Window."
        )

    mock_llm = RunnableLambda(mock_llm_invoke)

    # --- 2.2 MOCK RETRIEVER (Chặn không cho gọi FAISS/BM25 thật) ---
    mock_retriever = MagicMock()
    # Giả lập kết quả trả về khi tìm kiếm
    mock_retriever.invoke.return_value = [
        Document(
            page_content="SmartDoc hỗ trợ thuật toán Re-ranking.",
            metadata={"source": "bao_cao.pdf", "page": 1},
        )
    ]
    # Ép hàm setup_hybrid_retriever trả về cái retriever giả này
    mock_setup_retriever.return_value = mock_retriever

    # --- 2.3 CHẠY THỬ PIPELINE ---
    mock_vector_store = MagicMock()  # FAISS giả
    mock_docs = []  # Danh sách tài liệu gốc (không cần vì đã mock retriever)
    test_query = "Hệ thống SmartDoc dùng thuật toán gì và khác gì với hệ thống CoRAG?"

    # Truyền đúng 4 tham số bắt buộc theo chữ ký của hàm (signature)
    final_output = run_advanced_rag_pipeline(
        user_question=test_query,
        vector_store=mock_vector_store,
        documents=mock_docs,
        llm=mock_llm,
    )

    # --- 2.4 KHẲNG ĐỊNH KẾT QUẢ (ASSERTS) ---
    # Phải có đủ các keys do hệ thống quy định
    assert "answer" in final_output
    assert "confidence_score" in final_output
    assert "citations" in final_output
    assert "suggested_questions" in final_output

    # Kiểm tra tính chính xác của dữ liệu được nhào nặn qua Pipeline
    assert "Sliding Window" in final_output["answer"]
    assert final_output["confidence_score"] == 0.96
    assert len(final_output["citations"]) > 0
    assert final_output["citations"][0]["source"] == "bao_cao.pdf"
