from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.llm_provider import get_llm


def run_chat(query: str, chat_history: str):
    llm = get_llm()

    prompt = PromptTemplate(
        template="""
Bạn là SmartDoc AI trợ lý AI chuyên nghiệp trong hệ thống quản lý tài liệu (RAG system).

NHIỆM VỤ:
- Trò chuyện tự nhiên với người dùng
- Duy trì ngữ cảnh hội thoại
- Hỗ trợ về tài liệu nếu cần

NGUYÊN TẮC BẮT BUỘC:
1. Trả lời đúng ngôn ngữ của người dùng
2. Chỉ dựa vào "Lịch sử trò chuyện" để hiểu ngữ cảnh
3. KHÔNG bịa thông tin nếu không có trong lịch sử
4. Nếu không đủ thông tin → hỏi lại người dùng
5. Trả lời rõ ràng, tự nhiên, giống con người

💬 HÀNH VI:
- Nếu người dùng chào → chào lại + hỏi nhu cầu
- Nếu hội thoại đang tiếp diễn → trả lời tiếp mạch
- Nếu câu hỏi mơ hồ → hỏi lại

====================
📚 LỊCH SỬ:
{history}

👤 NGƯỜI DÙNG:
{query}

🤖 SMARTDOC AI:
""",
        input_variables=["history", "query"],
    )

    chain = prompt | llm | StrOutputParser()

    return chain.stream({"history": chat_history, "query": query})
