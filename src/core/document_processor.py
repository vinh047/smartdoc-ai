import os
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import TEXT_SPLITTER_CONFIG, EMBEDDING_CONFIG

def process_document(temp_path):
    # Trích xuất đuôi file
    file_extension = os.path.splitext(temp_path)[1].lower()
    
    # Chọn Loader phù hợp
    if file_extension == '.pdf':
        loader = PDFPlumberLoader(temp_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(temp_path)
    else:
        raise ValueError("Chỉ hỗ trợ file PDF và DOCX")
    
    # Đọc tài liệu
    docs = loader.load()

    # Chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"]  
    )
    documents = text_splitter.split_documents(docs)

    # Khởi tạo mô hình Embedding
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={'device': EMBEDDING_CONFIG["device"]},
        encode_kwargs={'normalize_embeddings': EMBEDDING_CONFIG["normalize_embeddings"]}
    )

    # Lưu vào FAISS Vector Store
    vector_store = FAISS.from_documents(documents, embedder)
    
    return vector_store