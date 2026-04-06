import os
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from src.config.settings import EMBEDDING_CONFIG, FAISS_INDEX_PATH
from src.core.database import SessionLocal, Document, DocumentChunk


def get_embedder():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={"device": EMBEDDING_CONFIG["device"]},
        encode_kwargs={
            "normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"]
        },
    )


def process_document(
    file_path: str, file_name: str, chunk_size: int, chunk_overlap: int
):
    """
    Xử lý file (Hỗ trợ Câu 1: DOCX, Câu 5: Source Tracking)
    """
    try:
        # Lựa chọn Loader dựa trên đuôi file
        if file_name.lower().endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return False

        docs = loader.load()

        # Chia nhỏ tài liệu
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(docs)

        # Lưu dữ liệu thô vào SQLite (Để phục vụ cho Summarize Map-Reduce và Source Citation)
        db = SessionLocal()
        db_doc = Document(file_name=file_name, file_path=file_path)
        db.add(db_doc)
        db.flush()  # Lấy ID

        for i, split in enumerate(splits):
            # Cập nhật metadata cho FAISS & BM25
            split.metadata["source"] = file_name
            split.metadata["chunk_id"] = f"{db_doc.id}_{i}"

            db_chunk = DocumentChunk(
                document_id=db_doc.id,
                chunk_index=i,
                content=split.page_content,
                page_number=split.metadata.get("page", 0),
            )
            db.add(db_chunk)

        db.commit()
        db.close()

        # Thêm vào FAISS (Vector DB)
        embedder = get_embedder()
        if os.path.exists(FAISS_INDEX_PATH):
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True
            )
            vector_store.add_documents(splits)
        else:
            vector_store = FAISS.from_documents(splits, embedder)
        vector_store.save_local(FAISS_INDEX_PATH)

        return True
    except Exception as e:
        print(f"Lỗi xử lý file: {e}")
        return False


def load_stores():
    """Tải FAISS và tạo BM25 index từ SQLite để phục vụ Hybrid Search (Câu 7)"""
    vector_store = None
    bm25_retriever = None

    if os.path.exists(FAISS_INDEX_PATH):
        embedder = get_embedder()
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True
        )

    # Tạo lại BM25 từ SQLite (BM25 chạy trên RAM nên cần load lại)
    db = SessionLocal()
    chunks = db.query(DocumentChunk).join(Document).all()
    if chunks:
        from langchain_core.documents import Document as LC_Document

        lc_docs = [
            LC_Document(
                page_content=c.content,
                metadata={
                    "source": c.document.file_name,
                    "page": c.page_number,
                    "chunk_id": f"{c.document_id}_{c.chunk_index}",
                },
            )
            for c in chunks
        ]
        bm25_retriever = BM25Retriever.from_documents(lc_docs)
        bm25_retriever.k = 10
    db.close()

    return vector_store, bm25_retriever
