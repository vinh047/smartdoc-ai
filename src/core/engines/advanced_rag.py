import json
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.core.rag_engine import setup_hybrid_retriever
from src.config import LLM_CONFIG, RETRIEVER_CONFIG

# Khởi tạo mô hình Cross-Encoder (Dùng để chấm điểm lại - Reranking)
try:
    cross_encoder = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512, device="cuda"
    )
except Exception as e:
    print(f"⚠️ Cảnh báo: Lỗi khởi tạo Cross-Encoder: {e}")
    cross_encoder = None


def decompose_query(query: str, llm: Any, chat_history: str = "") -> List[str]:
    """
    Query Rewriting: Sử dụng LLM và lịch sử chat để viết lại/tách nhỏ câu hỏi của người dùng,
    giúp hệ thống tìm kiếm hiểu rõ ngữ cảnh hơn.

    Args:
        query (str): Câu hỏi gốc của người dùng.
        llm (Any): Đối tượng LLM (Ollama).
        chat_history (str): Lịch sử trò chuyện gần nhất.

    Returns:
        List[str]: Danh sách các câu hỏi đã được làm rõ nghĩa.
    """
    decompose_template = """Bạn là chuyên gia phân tích truy vấn.
        Dựa vào lịch sử trò chuyện, hãy viết lại câu hỏi của người dùng cho thật rõ nghĩa.
        CHỈ TRẢ VỀ MẢNG JSON CÁC CHUỖI TEXT. Không giải thích.

        Lịch sử trò chuyện:
        {history}

        Câu hỏi gốc: {query}
        JSON Output:"""

    try:
        chain = (
            PromptTemplate(
                template=decompose_template, input_variables=["history", "query"]
            )
            | llm
        )
        result_raw = chain.invoke({"history": chat_history, "query": query})
        result_text = (
            result_raw.content if hasattr(result_raw, "content") else str(result_raw)
        )

        # Trích xuất JSON bằng Regex để chống lỗi LLM sinh ra text thừa
        match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
        else:
            parsed_json = json.loads(result_text)

        # Xử lý an toàn: Nếu LLM trả về Dict thay vì List
        if isinstance(parsed_json, dict):
            temp_list = []
            for v in parsed_json.values():
                if isinstance(v, list):
                    temp_list.extend(v)
                else:
                    temp_list.append(v)
            parsed_json = temp_list if temp_list else [query]
        elif not isinstance(parsed_json, list):
            parsed_json = [query]

        # Đảm bảo mọi phần tử đều là chuỗi (string)
        return [
            str(list(item.values())[0]) if isinstance(item, dict) else str(item)
            for item in parsed_json
        ]

    except Exception as e:
        print(f"⚠️ Decompose Fallback (Lỗi viết lại câu hỏi): {e}")
        return [str(query)]


def rerank_documents(
    query: str, docs: List[Document], top_k: int = 3
) -> List[Document]:
    """
    Chấm điểm lại (Rerank) độ liên quan giữa câu hỏi và danh sách tài liệu
    bằng mô hình Cross-Encoder chuyên dụng.

    Args:
        query (str): Câu hỏi đã được làm rõ.
        docs (List[Document]): Danh sách tài liệu lấy từ FAISS/BM25.
        top_k (int): Số lượng tài liệu tốt nhất cần giữ lại.

    Returns:
        List[Document]: Danh sách tài liệu đã được sắp xếp lại.
    """
    if not docs or cross_encoder is None:
        return docs[:top_k]

    # Ép kiểu an toàn để tránh lỗi Tokenizer của thư viện
    safe_query = str(query)
    pairs = [[safe_query, str(doc.page_content)] for doc in docs]

    # Chấm điểm
    scores = cross_encoder.predict(pairs)
    doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    reranked_docs = []
    for doc, score in doc_score_pairs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    return reranked_docs


def evaluate_with_self_rag(
    query: str, context: str, answer: str, llm: Any
) -> Dict[str, Any]:
    """
    Self-RAG: AI tự đóng vai trò giám khảo để chấm điểm câu trả lời của chính nó
    và đề xuất các hướng khám phá tài liệu tiếp theo.

    Returns:
        Dict: Chứa 'confidence_score' và 'suggested_questions'.
    """
    eval_template = """Bạn là hệ thống đánh giá Self-RAG. 
        Dựa vào ngữ cảnh, câu hỏi và câu trả lời dưới đây, hãy chấm điểm độ tự tin (0.0 đến 1.0) và gợi ý 2 câu hỏi tiếp theo.
        CHỈ TRẢ VỀ JSON BẮT BUỘC THEO ĐÚNG ĐỊNH DẠNG NÀY, KHÔNG GIẢI THÍCH GÌ THÊM:
        {{
            "confidence_score": 0.92,
            "suggested_questions": ["câu hỏi 1?", "câu hỏi 2?"]
        }}

        Ngữ cảnh: {context}
        Câu hỏi: {query}
        Câu trả lời AI: {answer}
        JSON Output:"""

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
            "suggested_questions": [
                "Bạn muốn hỏi thêm gì về chi tiết trong tài liệu này?"
            ],
        }


def run_advanced_rag_pipeline(
    user_question: str,
    vector_store: Any,
    documents: List[Document],
    llm: Any,
    chat_history: str = "",
    search_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Pipeline chính của Advanced CoRAG (Giai đoạn 3).
    Quy trình: Rewrite -> Hybrid Retrieve -> Cross-Encoder Rerank -> Generate -> Self-RAG.

    Returns:
        Dict: Chứa answer, confidence_score, citations, suggested_questions.
    """
    # 1. Viết lại và tách câu hỏi (Query Rewriting)
    sub_queries = decompose_query(user_question, llm, chat_history)
    print(
        f"🚀 [AI Rewriting] Đã tách thành {len(sub_queries)} sub-queries: {sub_queries}"
    )

    # 2. Truy xuất và Chấm điểm lại ĐỘC LẬP (Retrieve & Rerank)
    retriever = setup_hybrid_retriever(vector_store, documents, search_filter)
    unique_top_docs = {}

    for sq in sub_queries:
        sq_initial_docs = retriever.invoke(sq)
        sq_top_docs = rerank_documents(sq, sq_initial_docs, top_k=4)

        # Loại bỏ các đoạn tài liệu trùng lặp
        for doc in sq_top_docs:
            if doc.page_content not in unique_top_docs:
                unique_top_docs[doc.page_content] = doc

    top_docs = list(unique_top_docs.values())

    # Xử lý trường hợp không tìm thấy tài liệu
    if not top_docs:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong tài liệu hệ thống.",
            "confidence_score": 0.0,
            "citations": [],
            "suggested_questions": [],
        }

    # 3. Chuẩn bị ngữ cảnh và trích dẫn
    context_text = "\n\n".join([doc.page_content for doc in top_docs])
    citations = [
        {
            "source": doc.metadata.get("source", "unknown.pdf"),
            "page": doc.metadata.get("page", "N/A"),
            "hybrid_score": doc.metadata.get("score", 0.0),
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "content": doc.page_content,  # <--- THÊM DÒNG NÀY ĐỂ LẤY FULL TEXT
        }
        for doc in top_docs
    ]

    # 4. Sinh câu trả lời (Generation)
    main_template = """Bạn là chuyên gia phân tích tài liệu. Nhiệm vụ của bạn là trả lời câu hỏi DỰA HOÀN TOÀN vào "Tài liệu trích xuất" bên dưới.

        Quy tắc TỐI THƯỢNG:
        1. TRẢ LỜI 100% BẰNG TIẾNG VIỆT.
        2. Không tự ý bịa đặt thông tin ngoài tài liệu.

        Lịch sử trò chuyện:
        {chat_history}

        Tài liệu trích xuất: 
        {context}

        Câu hỏi: {question}
        Câu trả lời:"""

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

    # 5. Đánh giá chất lượng (Self-RAG)
    self_rag_data = evaluate_with_self_rag(user_question, context_text, answer, llm)

    return {
        "answer": answer,
        "confidence_score": float(self_rag_data.get("confidence_score", 0.85)),
        "citations": citations,
        "suggested_questions": self_rag_data.get("suggested_questions", []),
    }
