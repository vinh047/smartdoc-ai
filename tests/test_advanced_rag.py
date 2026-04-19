import json
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from src.core.engines.advanced_rag import run_advanced_rag_pipeline


def test_mock_advanced_rag():
    print("\n🚀 [GIAI ĐOẠN 1] CHẠY KIỂM THỬ MOCK DATA CHO TV5")
    print("-" * 60)

    # --- MOCK LLM (Giả lập bằng RunnableLambda để tương thích LCEL LangChain) ---
    def mock_llm_invoke(prompt_val):
        prompt_str = str(prompt_val)

        # 1. Bắt ngữ cảnh của Prompt chia nhỏ câu hỏi (Multi-hop)
        if "CHỈ TRẢ VỀ MẢNG JSON" in prompt_str:
            return '["SmartDoc dùng thuật toán gì?", "CoRAG là gì?"]'

        # 2. Bắt ngữ cảnh của Prompt Self-RAG
        if "confidence_score" in prompt_str:
            return '{"confidence_score": 0.96, "suggested_questions": ["Tại sao cần Re-ranking?", "RAG khác gì CoRAG?"]}'

        # 3. Trả về đối tượng mock giả lập AIMessage cho bước Generate chính
        class FakeAIMessage:
            def __init__(self, content):
                self.content = content

        return FakeAIMessage(
            "SmartDoc dùng Cross-Encoder để Re-ranking. Khác với SmartDoc, CoRAG dùng thêm bộ nhớ Sliding Window cho hội thoại."
        )

    mock_llm = RunnableLambda(mock_llm_invoke)

    # --- MOCK DOCS (Giả lập tập tài liệu thô TV3 đưa qua, chứa thông tin rải rác) ---
    mock_initial_docs = [
        Document(
            page_content="SmartDoc hỗ trợ thuật toán Re-ranking để tăng độ chính xác tìm kiếm.",
            metadata={"source": "bao_cao.pdf", "page": 0, "score": 0.88},
        ),
        Document(
            page_content="Hôm nay trời mưa khá to tại TP.HCM.",
            metadata={"source": "thoi_tiet.pdf", "page": 5, "score": 0.45},
        ),
        Document(
            page_content="CoRAG (Conversational RAG) là hệ thống có khả năng nhớ lịch sử nhờ Sliding Window.",
            metadata={"source": "bao_cao.pdf", "page": 1, "score": 0.81},
        ),
        Document(
            page_content="React là một thư viện Javascript phổ biến.",
            metadata={"source": "IT_basic.pdf", "page": 2, "score": 0.30},
        ),
    ]

    # --- CHẠY THỬ PIPELINE MULTI-HOP ---
    # Câu hỏi phức tạp đòi hỏi phải chia làm 2 ý để truy xuất
    test_query = "Hệ thống SmartDoc dùng thuật toán gì và khác gì với hệ thống CoRAG?"
    print(f"👉 Câu hỏi Test: {test_query}\n")

    final_output = run_advanced_rag_pipeline(test_query, mock_initial_docs, mock_llm)

    print("\n✅OUTPUT(TV5 gửi cho TV1):")
    print(json.dumps(final_output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    test_mock_advanced_rag()
