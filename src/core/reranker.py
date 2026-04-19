from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Sắp xếp lại kết quả tìm kiếm bằng mô hình Cross-Encoder
    để cải thiện độ chính xác của RAG pipeline.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Khởi tạo Cross-Encoder model.

        Args:
            model_name: Tên mô hình Cross-Encoder từ HuggingFace.
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 3
    ) -> List[Document]:
        """
        Chấm điểm và sắp xếp lại danh sách tài liệu dựa trên độ liên quan với query.

        Args:
            query: Câu hỏi của người dùng.
            documents: Danh sách LangChain Document objects cần rerank.
            top_k: Số lượng tài liệu tốt nhất cần trả về.

        Returns:
            Danh sách top_k Document đã được sắp xếp theo độ liên quan giảm dần.
        """
        if not documents:
            return []

        # Tạo cặp (query, document_content) để cross-encoder chấm điểm
        pairs = [(query, doc.page_content) for doc in documents]

        # Dự đoán điểm số liên quan cho từng cặp
        scores = self.model.predict(pairs)

        # Gắn điểm số vào từng document để tiện debug/logging
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        # Sắp xếp theo điểm giảm dần và lấy top_k
        sorted_docs = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        return [doc for _, doc in sorted_docs[:top_k]]
