import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from sentence_transformers import CrossEncoder

from src.config.settings import LLM_CONFIG, RETRIEVER_CONFIG, RERANKER_CONFIG
from src.core.database import log_retrieval

# 1. Khởi tạo LLM
llm = OllamaLLM(
    model=LLM_CONFIG["model"],
    temperature=LLM_CONFIG["temperature"],
    top_p=LLM_CONFIG["top_p"],
    repeat_penalty=LLM_CONFIG["repeat_penalty"],
)


# 2. Khởi tạo Reranker NATIVE (Chỉ load 1 lần)
@st.cache_resource(show_spinner=False)
def get_reranker():
    # Gọi trực tiếp model lõi, không qua Langchain wrapper
    return CrossEncoder(RERANKER_CONFIG["model_name"])


def rewrite_query(user_question, chat_history):
    if not chat_history.strip():
        return user_question
    prompt = PromptTemplate(
        template="""Dựa vào lịch sử, hãy viết lại câu hỏi cuối thành một câu hỏi đầy đủ ngữ cảnh độc lập.
        CHỈ IN RA CÂU HỎI. KHÔNG TRẢ LỜI.
        Lịch sử:\n{chat_history}\nCâu hỏi: {question}\nCâu độc lập:""",
        input_variables=["chat_history", "question"],
    )
    return (
        (prompt | llm | StrOutputParser())
        .invoke({"chat_history": chat_history, "question": user_question})
        .strip()
    )


def evaluate_relevance(query, context):
    if not context:
        return False
    prompt = PromptTemplate(
        template="""Đánh giá xem tài liệu có chứa thông tin trả lời câu hỏi không.
        Tài liệu: {context}
        Câu hỏi: {question}
        Trả lời duy nhất 'YES' (có) hoặc 'NO' (không).""",
        input_variables=["context", "question"],
    )
    result = (
        (prompt | llm | StrOutputParser())
        .invoke({"context": context, "question": query})
        .strip()
        .lower()
    )
    return any(word in result for word in ["yes", "có", "true"])


def web_search_fallback(query):
    search = DuckDuckGoSearchRun()
    try:
        results = search.invoke(query)
        return results
    except Exception as e:
        print(e)
        return ""


def run_rag_chain(
    user_question,
    vector_store,
    bm25_retriever,
    session_id,
    chat_history="",
    selected_files=None,
):
    # 1. CoRAG - Viết lại câu hỏi
    standalone_query = rewrite_query(user_question, chat_history)

    source_docs = []
    meta_info = {
        "standalone_query": standalone_query,
        "is_relevant": False,
        "used_web": False,
    }

    if vector_store and bm25_retriever:
        # 2. Native Hybrid Search (FAISS + BM25)
        # Lấy kết quả từ FAISS
        faiss_retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_CONFIG["k_fetch"]}
        )
        faiss_docs = faiss_retriever.invoke(standalone_query)

        # Lấy kết quả từ BM25
        bm25_retriever.k = RETRIEVER_CONFIG["k_fetch"]
        bm25_docs = bm25_retriever.invoke(standalone_query)

        # Gộp tài liệu và xóa trùng lặp (Deduplicate)
        unique_docs = {}
        for doc in faiss_docs + bm25_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in unique_docs:
                unique_docs[chunk_id] = doc

        raw_docs = list(unique_docs.values())

        # 3. Metadata Filtering (Lọc tài liệu theo UI)
        if selected_files:
            raw_docs = [
                doc for doc in raw_docs if doc.metadata.get("source") in selected_files
            ]

        # 4. Native Re-ranking bằng Cross-Encoder
        if raw_docs:
            reranker = get_reranker()
            # Tạo cặp [câu hỏi, đoạn văn] để model chấm điểm
            pairs = [[standalone_query, doc.page_content] for doc in raw_docs]
            scores = reranker.predict(pairs)

            # Sắp xếp tài liệu theo điểm số giảm dần
            scored_docs = sorted(
                zip(raw_docs, scores), key=lambda x: x[1], reverse=True
            )

            # Chỉ lấy Top N tài liệu có điểm cao nhất
            source_docs = [
                doc for doc, score in scored_docs[: RERANKER_CONFIG["top_n"]]
            ]

    context_text = "\n\n".join(doc.page_content for doc in source_docs)
    chunk_ids = [doc.metadata.get("chunk_id", "unknown") for doc in source_docs]

    # 5. CRAG Evaluator (Self-RAG)
    is_relevant = evaluate_relevance(standalone_query, context_text)
    meta_info["is_relevant"] = is_relevant

    # 6. Fallback Web Search (Self-RAG Cấp cao)
    if not is_relevant:
        web_context = web_search_fallback(standalone_query)
        if web_context:
            meta_info["used_web"] = True
            context_text = web_context
            fallback_template = """Ngữ cảnh tài liệu local KHÔNG có thông tin này. Tuy nhiên, tôi đã tìm kiếm trên Web (DuckDuckGo).
            Dựa vào kết quả Web sau đây, hãy trả lời câu hỏi:

            Kết quả Web: {context}
            Câu hỏi: {question}
            Trả lời:"""
            prompt = PromptTemplate(
                template=fallback_template, input_variables=["context", "question"]
            )
            chain = prompt | llm | StrOutputParser()
            return (
                chain.stream({"context": context_text, "question": standalone_query}),
                [],
                meta_info,
            )
        else:
            # Fallback chat cơ bản
            fallback_prompt = PromptTemplate(
                template="Ngữ cảnh: KHÔNG CÓ THÔNG TIN. Hỏi: {question}. Hãy trả lời dựa trên kiến thức chung hoặc nói không biết lịch sự. Trả lời:",
                input_variables=["question"],
            )
            return (
                (fallback_prompt | llm | StrOutputParser()).stream(
                    {"question": standalone_query}
                ),
                [],
                meta_info,
            )

    log_retrieval(
        session_id,
        standalone_query,
        chunk_ids,
        {"crag_evaluator": is_relevant},
        is_relevant,
        not is_relevant,
    )

    # 7. Final Generation (RAG Chuẩn) - CẬP NHẬT PROMPT SIÊU CẤP
    qa_template = """Bạn là SmartDoc AI, một chuyên gia phân tích tài liệu và trợ lý học tập thân thiện.
    Dựa vào các thông tin được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách mạch lạc, dễ hiểu nhất.

    🔥 YÊU CẦU TRÌNH BÀY (BẮT BUỘC):
    1. Trình bày rõ ràng: KHÔNG copy y nguyên từng câu cứng nhắc. Hãy diễn đạt lại cho mượt mà nhưng vẫn giữ nguyên tính chính xác của học thuật.
    2. Sử dụng Markdown: Bắt buộc dùng in đậm (**từ khóa**) cho các thuật ngữ quan trọng.
    3. Cấu trúc cấu trúc: Nếu nội dung có nhiều ý (ví dụ: các phương pháp, các tính chất), bắt buộc phải dùng gạch đầu dòng (-) hoặc đánh số (1, 2, 3) để phân chia.
    4. Thêm lời dẫn: Bắt đầu bằng một câu dẫn nhập thân thiện và kết luận gọn gàng.

    Ngữ cảnh từ tài liệu:
    {context}

    Câu hỏi của người dùng: {question}

    Câu trả lời của bạn:"""

    chain = (
        PromptTemplate(template=qa_template, input_variables=["context", "question"])
        | llm
        | StrOutputParser()
    )
    return (
        chain.stream({"context": context_text, "question": standalone_query}),
        source_docs,
        meta_info,
    )
