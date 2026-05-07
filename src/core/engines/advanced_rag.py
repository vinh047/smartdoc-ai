import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.core.rag_engine import setup_hybrid_retriever

# TỐI ƯU 1: Lazy Loading & Singleton cho Cross-Encoder
_cross_encoder_instance = None


def get_cross_encoder():
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        try:
            print("🚀 Đang khởi tạo mô hình Cross-Encoder...")
            _cross_encoder_instance = CrossEncoder(
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                max_length=512,
                device="cuda",  # Chuyển thành "cpu" nếu không có GPU
            )
        except Exception as e:
            print(f"⚠️ Cảnh báo: Lỗi khởi tạo Cross-Encoder: {e}")
    return _cross_encoder_instance


def decompose_query(query: str, llm: Any, chat_history: str = "") -> List[str]:
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

        match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
        else:
            parsed_json = json.loads(result_text)

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
    encoder = get_cross_encoder()
    if not docs or encoder is None:
        return docs[:top_k]

    safe_query = str(query)
    pairs = [[safe_query, str(doc.page_content)] for doc in docs]

    scores = encoder.predict(pairs)
    doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    reranked_docs = []
    for doc, score in doc_score_pairs[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    return reranked_docs


def evaluate_with_self_rag(
    query: str, context: str, answer: str, llm: Any
) -> Dict[str, Any]:
    # 1. Phát hiện ngôn ngữ dựa trên câu hỏi của người dùng
    vietnamese_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
    is_vietnamese = any(char in query.lower() for char in vietnamese_chars)

    # 2. Thiết lập Template dựa trên ngôn ngữ phát hiện được
    if is_vietnamese:
        eval_template = """Bạn là hệ thống đánh giá Self-RAG. 
        Nhiệm vụ: Chấm điểm độ tự tin và gợi ý 2 câu hỏi tiếp theo BẰNG TIẾNG VIỆT.
        CHỈ TRẢ VỀ JSON THEO ĐỊNH DẠNG:
        {{
            "confidence_score": 0.92,
            "suggested_questions": ["câu hỏi 1?", "câu hỏi 2?"]
        }}
        Ngữ cảnh: {context}
        Câu hỏi: {query}
        Trả lời: {answer}
        JSON Output:"""
    else:
        eval_template = """You are a Self-RAG evaluation system.
        Task: Score the confidence and suggest 2 follow-up questions IN ENGLISH.
        ONLY RETURN JSON IN THIS FORMAT:
        {{
            "confidence_score": 0.92,
            "suggested_questions": ["question 1?", "question 2?"]
        }}
        Context: {context}
        Question: {query}
        Answer: {answer}
        JSON Output:"""

    try:
        # 3. Thực thi Chain (giữ nguyên logic invoke và xử lý Regex cũ của bạn)
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
        return {"confidence_score": 0.0, "suggested_questions": [], "eval_error": True}


def fetch_and_rerank_task(sq: str, retriever: Any, top_k: int = 4) -> List[Document]:
    try:
        sq_initial_docs = retriever.invoke(sq)
        return rerank_documents(sq, sq_initial_docs, top_k=top_k)
    except Exception as e:
        print(f"⚠️ Lỗi truy xuất cho query '{sq}': {e}")
        return []


def run_advanced_rag_pipeline(
    user_question: str,
    vector_store: Any,
    documents: List[Document],
    llm: Any,
    chat_history: str = "",
    search_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # 1. Viết lại và tách câu hỏi
    sub_queries = decompose_query(user_question, llm, chat_history)
    
    print(f"🔍 [Query Decomposition] Gốc: {user_question}")
    print(f"🔍 [Query Decomposition] Tách ra: {sub_queries}")
    
    if not sub_queries:
        sub_queries = [user_question]

    # 2. Truy xuất và Chấm điểm lại ĐỒNG THỜI
    retriever = setup_hybrid_retriever(vector_store, documents, search_filter)
    unique_top_docs = {}

    with ThreadPoolExecutor(max_workers = max(min(len(sub_queries), 5), 1)) as executor:
        future_to_sq = {
            executor.submit(fetch_and_rerank_task, sq, retriever): sq
            for sq in sub_queries
        }
        for future in as_completed(future_to_sq):
            sq_top_docs = future.result()
            for doc in sq_top_docs:
                if doc.page_content not in unique_top_docs:
                    unique_top_docs[doc.page_content] = doc

    top_docs = list(unique_top_docs.values())

    if not top_docs:
        # Nếu không có docs, trả về object rỗng để UI tự xử lý
        return {
            "answer_stream": (
                chunk
                for chunk in [
                    "Không tìm thấy thông tin phù hợp trong tài liệu hệ thống."
                ]
            ),
            "context_text": "",
            "citations": [],
        }

    top_docs.sort(key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
    top_docs = top_docs[:5]

    # 3. Chuẩn bị ngữ cảnh và trích dẫn
    context_text = "\n\n".join([doc.page_content for doc in top_docs])
    citations = [
        {
            "source": doc.metadata.get("source", "unknown.pdf"),
            "page": doc.metadata.get("page", "N/A"),
            "hybrid_score": doc.metadata.get("score", 0.0),
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "content": doc.page_content,
        }
        for doc in top_docs
    ]

    # 4. Sinh câu trả lời (SỬ DỤNG STREAMING THAY VÌ INVOKE)
    vietnamese_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
    is_vietnamese = any(char in user_question.lower() for char in vietnamese_chars)

    # 2. Định nghĩa bộ Template song ngữ
    if is_vietnamese:
        main_template = """Bạn là chuyên gia phân tích tài liệu. Trả lời câu hỏi DỰA HOÀN TOÀN vào "Ngữ cảnh trích xuất" bên dưới.
        Nếu thông tin không có trong ngữ cảnh, hãy nói bạn không biết.
        
        Quy tắc:
        1. TRẢ LỜI 100% BẰNG TIẾNG VIỆT.
        2. Không tự ý bịa đặt thông tin.

        Lịch sử: {chat_history}
        Ngữ cảnh: {context}
        Câu hỏi: {question}
        Trả lời:"""
    else:
        main_template = """You are a document analysis expert. Answer the question based SOLELY on the "Extracted Context" below.
        If the information is not in the context, simply say you don't know.
        
        Rules:
        1. YOU MUST ANSWER IN ENGLISH.
        2. Do not invent information outside the context.

        History: {chat_history}
        Context: {context}
        Question: {question}
        Answer:"""

    chain = (
        PromptTemplate(
            template=main_template,
            input_variables=["context", "question", "chat_history"],
        )
        | llm
    )

    # GỌI STREAM Thay vì invoke để nhả từng chữ
    answer_stream = chain.stream(
        {
            "context": context_text,
            "question": user_question,
            "chat_history": chat_history,
        }
    )

    # TRẢ VỀ LUỒNG (STREAM) và CONTEXT ĐỂ UI CHẠY SELF-RAG SAU
    return {
        "answer_stream": answer_stream,
        "context_text": context_text,
        "citations": citations,
    }
