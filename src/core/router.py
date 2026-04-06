from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm_provider import get_llm


def classify_intent(query: str, chat_history: str) -> str:
    """Phân tích 4 tin nhắn gần nhất và câu hỏi để xác định Intent. Cực kỳ tốc độ."""
    llm = get_llm()

    prompt = PromptTemplate(
        template="""Nhiệm vụ: Phân loại ý định của người dùng vào ĐÚNG MỘT TỪ trong danh sách sau:
        [CHAT, RAG, SUMMARIZE, COMPARE, EXTRACT, TRANSLATE, WEB, EXPLAIN]

        - CHAT: Giao tiếp xã giao, thông tin của bạn, chào hỏi, cảm ơn.
        - SUMMARIZE: Yêu cầu tóm tắt toàn bộ tài liệu, chương, file.
        - RAG: Tìm kiếm thông tin, kiến thức chi tiết trong tài liệu.
        - COMPARE: So sánh điểm giống/khác nhau giữa 2 đối tượng/tài liệu.
        - EXTRACT: Yêu cầu trích xuất danh sách, lập bảng, lấy số liệu.
        - TRANSLATE: Dịch thuật tài liệu sang ngôn ngữ khác.
        - WEB: Hỏi thông tin thời sự, giá cả, thời tiết, tin tức mới nhất (không có trong sách).
        - EXPLAIN: Giải thích một khái niệm, thuật ngữ chuyên ngành.

        Lịch sử trò chuyện:
        {history}

        Câu hỏi hiện tại: "{query}"

        TỪ KHÓA Ý ĐỊNH (Chỉ in ra 1 từ duy nhất):""",
        input_variables=["history", "query"],
    )

    # Ép LLM trả về đúng 1 từ và format lại
    intent = (prompt | llm | StrOutputParser()).invoke(
        {"history": chat_history, "query": query}
    )

    valid_intents = [
        "CHAT",
        "RAG",
        "SUMMARIZE",
        "COMPARE",
        "EXTRACT",
        "TRANSLATE",
        "WEB",
        "EXPLAIN",
    ]
    cleaned_intent = "".join(filter(str.isalpha, intent.strip().upper()))

    for valid in valid_intents:
        if valid in cleaned_intent:
            return valid

    return "RAG"  # Fallback mặc định an toàn nhất
