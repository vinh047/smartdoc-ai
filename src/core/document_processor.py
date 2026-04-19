import os
from typing import List

from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import TEXT_SPLITTER_CONFIG, EMBEDDING_CONFIG
from src.core.metadata_handler import MetadataManager


def process_document(temp_path: str, filename: str) -> List[Document]:
    """
    Đọc, chia nhỏ và gắn metadata cho một tài liệu.
    Trả về danh sách các Document chunks để app.py có thể gộp
    nhiều file trước khi tạo FAISS vector store duy nhất.

    Args:
        temp_path: Đường dẫn tạm thời của file đã được lưu trên đĩa.
        filename: Tên file gốc (dùng để tạo metadata 'source').

    Returns:
        Danh sách LangChain Document objects đã được chia nhỏ và gắn metadata.

    Raises:
        ValueError: Nếu định dạng file không được hỗ trợ.
    """
    # 1. Xác định định dạng và chọn Loader phù hợp
    file_extension = os.path.splitext(temp_path)[1].lower()

    if file_extension == ".pdf":
        loader = PDFPlumberLoader(temp_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(temp_path)
    else:
        raise ValueError(f"Định dạng '{file_extension}' không được hỗ trợ. Chỉ chấp nhận PDF và DOCX.")

    # 2. Đọc nội dung tài liệu
    raw_docs = loader.load()

    # 3. Chia nhỏ văn bản thành các chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
    )
    documents = text_splitter.split_documents(raw_docs)

    # 4. Tạo và gắn metadata cho toàn bộ chunks của file này
    metadata_manager = MetadataManager()

    # Xác định doc_type dựa theo extension
    doc_type = "pdf" if file_extension == ".pdf" else "docx"
    metadata = metadata_manager.create_metadata(filename=filename, doc_type=doc_type)

    documents = metadata_manager.attach_metadata_to_docs(documents, metadata)

    return documents