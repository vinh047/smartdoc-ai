import os

# ==========================================
# 0. Cấu hình Hệ thống & Lưu trữ
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = f"sqlite:///{os.path.join(DATA_DIR, 'smartdoc.db')}"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# ==========================================
# 1. Cấu hình Mặc định Xử lý tài liệu (Có thể ghi đè từ UI)
# ==========================================
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

# ==========================================
# 2. Cấu hình Embeddings & Re-ranking
# ==========================================
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "device": "cpu",
    "normalize_embeddings": True,
}

RERANKER_CONFIG = {
    # Dùng BAAI/bge-reranker-base cho đa ngôn ngữ cực tốt, hoặc ms-marco cho nhẹ
    "model_name": "BAAI/bge-reranker-base",
    "top_n": 3,  # Lấy 3 đoạn văn chuẩn nhất sau khi rerank
}

# ==========================================
# 3. Cấu hình Hybrid Search (BM25 + FAISS)
# ==========================================
RETRIEVER_CONFIG = {
    "search_type": "similarity",
    "k_fetch": 10,  # Lấy dư ra từ mỗi DB để Re-ranker lọc lại
    "faiss_weight": 0.5,
    "bm25_weight": 0.5,
}

# ==========================================
# 4. Cấu hình LLM
# ==========================================
LLM_CONFIG = {
    "model": "qwen2.5:7b",
    "temperature": 0.4,  # Rất thấp để tránh ảo giác (Hallucination)
    "top_p": 0.9,
    "repeat_penalty": 1.1,
}
