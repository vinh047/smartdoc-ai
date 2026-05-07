import os

# 1. Cấu hình xử lý tài liệu (Document Processing)
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,       # Kích thước tối đa mỗi đoạn
    "chunk_overlap": 100      # Số ký tự chồng lấp giữa các đoạn
}

# 2. Cấu hình mô hình nhúng (Embedding Model)
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Hỗ trợ tiếng Việt
    # "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # Hỗ trợ tiếng Việt
    "device": "cuda",          # Đổi thành 'cuda' nếu cậu dùng GPU có CUDA
    "normalize_embeddings": True
}

# 3. Cấu hình bộ máy tìm kiếm (Retriever)
RETRIEVER_CONFIG = {
    "search_type": "similarity", # Có thể đổi thành "mmr" để lấy kết quả đa dạng hơn
    "k": 3                      # Số lượng kết quả lấy ra (cậu đang dùng 10, mặc định dự án là 3)
}

# 4. Cấu hình mô hình ngôn ngữ lớn (LLM)
LLM_CONFIG = {
    "model": "qwen2.5:7b",       # Mô hình Qwen
    "temperature": 0.7,          # Độ sáng tạo
    "top_p": 0.9,                # Lấy mẫu Nucleus
    "repeat_penalty": 1.1        # Tránh lặp từ
}