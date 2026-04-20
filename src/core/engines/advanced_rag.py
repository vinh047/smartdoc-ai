import json
import re
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.config import LLM_CONFIG, RETRIEVER_CONFIG

try:
    cross_encoder = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512
    )
except Exception as e:
    print(f"⚠️ Cảnh báo: Lỗi khởi tạo Cross-Encoder: {e}")
    cross_encoder = None


def decompose_query(query: str, llm) -> list:
    """
    8.2.10: Query Rewriting & Multi-hop Reasoning.
    Chia nhỏ câu hỏi phức tạp thành các sub-queries tối ưu cho tìm kiếm.
    """
    decompose_template = """Bạn là chuyên gia phân tích truy vấn.
Hãy phân tích câu hỏi của người dùng và chia nó thành 1 hoặc nhiều câu hỏi nhỏ (sub-queries) rõ ràng hơn, thêm từ khóa nếu cần.
CHỈ TRẢ VỀ MẢNG JSON, KHÔNG GIẢI THÍCH GÌ THÊM.

Câu hỏi: {query}
JSON Output:"""

    try:
        chain = (
            PromptTemplate(template=decompose_template, input_variables=["query"]) | llm
        )
        result_raw = chain.invoke({"query": query})
        result_text = (
            result_raw.content if hasattr(result_raw, "content") else str(result_raw)
        )

        # Regex tìm mảng JSON
        match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(result_text)
    except Exception as e:
        print(f"⚠️ Decompose Query Error: {e}")
        return [query]  # Fallback: Giữ nguyên câu hỏi gốc nếu lỗi


def rerank_documents(query: str, docs: list, top_k: int = 3) -> list:
    """
    Chấm điểm lại (Re-ranking) danh sách chunk từ VectorDB dựa trên Cross-Encoder.
    """
    if not docs:
        return []
    if cross_encoder is None:
        return docs[:top_k]

    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)

    doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    reranked_docs = []
    for doc, score in doc_score_pairs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    return reranked_docs


def evaluate_with_self_rag(query: str, context: str, answer: str, llm) -> dict:
    """
    Self-RAG - AI tự đánh giá câu trả lời và sinh câu hỏi gợi ý.
    """
    eval_template = """Bạn là hệ thống đánh giá Self-RAG. 
Dựa vào ngữ cảnh, câu hỏi và câu trả lời dưới đây, hãy chấm điểm độ tự tin (0.0 đến 1.0) và gợi ý 2 câu hỏi tiếp theo.
CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH GÌ THÊM.

Ngữ cảnh: {context}
Câu hỏi: {query}
Câu trả lời AI: {answer}

Định dạng JSON BẮT BUỘC:
{{
    "confidence_score": 0.92,
    "suggested_questions": ["câu hỏi 1?", "câu hỏi 2?"]
}}"""

    try:
        chain = (
            PromptTemplate(
                template=eval_template, input_variables=["context", "query", "answer"]
            )
            | llm
        )
        result_raw = chain.invoke(
            {"context": context, "query": query, "answer": answer}
        )
        result_text = (
            result_raw.content if hasattr(result_raw, "content") else str(result_raw)
        )

        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(result_text)
    except Exception as e:
        print(f"⚠️ Self-RAG Error: {e}")
        return {
            "confidence_score": 0.85,
            "suggested_questions": ["Bạn muốn hỏi thêm gì về chủ đề này?"],
        }


# -----------------------------------------------------------------------------
# HÀM TỔNG HỢP CHO GIAI ĐOẠN 3 (TV1 GỌI - CHUẨN KẾT NỐI INTERFACE CONTRACT)
# -----------------------------------------------------------------------------
def run_advanced_rag_pipeline(
    user_question: str, initial_docs: list, llm, chat_history: str = ""
) -> dict:
    """
    Pipeline chính: Rewrite/Decompose -> Multi-hop Rerank -> Generate -> Self-RAG
    """
    if not initial_docs:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong tài liệu.",
            "confidence_score": 0.0,
            "citations": [],
            "suggested_questions": [],
        }

    # Bước 1: Query Rewriting & Decomposition (Phân rã câu hỏi)
    sub_queries = decompose_query(user_question, llm)
    print(
        f"[AI Rewriting] Đã tách thành {len(sub_queries)} sub-queries: {sub_queries}"
    )

    # Bước 2: Multi-hop Reranking (Duyệt từng sub-query để bóc tách chunk tốt nhất)
    unique_top_docs = {}
    for sq in sub_queries:
        sq_top_docs = rerank_documents(sq, initial_docs, top_k=2)
        for doc in sq_top_docs:
            # Dùng page_content làm key để loại bỏ chunk trùng lặp
            if doc.page_content not in unique_top_docs:
                unique_top_docs[doc.page_content] = doc

    top_docs = list(unique_top_docs.values())

    # Bước 3: Xây dựng Context và Citations
    context_text = "\n\n".join([doc.page_content for doc in top_docs])
    citations = [
        {
            "source": doc.metadata.get("source", "unknown.pdf"),
            "page": doc.metadata.get("page", 0) + 1,
            "hybrid_score": doc.metadata.get("score", 0.0),
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
        }
        for doc in top_docs
    ]

    # Bước 4: Sinh câu trả lời (Generation)
    main_template = """Sử dụng ngữ cảnh và lịch sử trò chuyện để trả lời câu hỏi bằng tiếng Việt đầy đủ và súc tích.
Lịch sử: {chat_history}
Ngữ cảnh: {context}
Câu hỏi gốc: {question}
Trả lời:"""

    answer_raw = (
        PromptTemplate(
            template=main_template,
            input_variables=["context", "question", "chat_history"],
        )
        | llm
    ).invoke(
        {
            "context": context_text,
            "question": user_question,
            "chat_history": chat_history,
        }
    )

    answer = answer_raw.content if hasattr(answer_raw, "content") else str(answer_raw)

    # Bước 5: Self-RAG đánh giá
    self_rag_data = evaluate_with_self_rag(user_question, context_text, answer, llm)

    return {
        "answer": answer,
        "confidence_score": float(self_rag_data.get("confidence_score", 0.85)),
        "citations": citations,
        "suggested_questions": self_rag_data.get("suggested_questions", []),
    }
