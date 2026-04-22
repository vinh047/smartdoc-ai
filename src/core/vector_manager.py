from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from src.config import EMBEDDING_CONFIG


def build_vector_store(chunks: list):
    """
    Chuyển đổi danh sách các đoạn văn bản (chunks) thành Vector Database FAISS.
    """
    if not chunks:
        raise ValueError("Không trích xuất được nội dung văn bản từ tài liệu.")

    # Ép mảng Dict về chuẩn Document của Langchain
    documents = [
        LangchainDocument(page_content=item["page_content"], metadata=item["metadata"])
        for item in chunks
    ]

    # Khởi tạo mô hình nhúng (Embedding)
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={"device": EMBEDDING_CONFIG["device"]},
        encode_kwargs={
            "normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"]
        },
    )

    # Đưa vào FAISS
    return FAISS.from_documents(documents, embedder)
