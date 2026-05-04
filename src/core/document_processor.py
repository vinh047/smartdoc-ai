import os
import pymupdf
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

from src.config import TEXT_SPLITTER_CONFIG

# Import các logic đã được tách riêng
from src.core.ocr_utils import (
    clean_ocr_text,
    is_text_good_enough,
    extract_text_with_ocr,
)

def process_document_data(temp_path: str, chunk_size=None, chunk_overlap=None, use_ocr=False):
    """
    Hàm định tuyến xử lý tùy theo đuôi file (PDF hoặc DOCX).
    Thêm tham số use_ocr để kiểm soát việc quét ảnh.
    """
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Không tìm thấy file: {temp_path}")

    c_size = chunk_size or TEXT_SPLITTER_CONFIG["chunk_size"]
    c_overlap = chunk_overlap or TEXT_SPLITTER_CONFIG["chunk_overlap"]

    file_ext = os.path.splitext(temp_path)[1].lower()
    if file_ext == ".pdf":
        # Truyền use_ocr vào hàm xử lý PDF
        return process_pdf(temp_path, c_size, c_overlap, use_ocr)
    elif file_ext == ".docx":
        return process_docx(temp_path, c_size, c_overlap)
    else:
        raise ValueError("Chỉ hỗ trợ file PDF và DOCX")


def process_pdf(file_path: str, chunk_size: int, chunk_overlap: int, use_ocr: bool = False):
    """
    Xử lý file PDF. Chỉ kích hoạt OCR khi use_ocr=True và văn bản trực tiếp không đạt yêu cầu.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_name = os.path.basename(file_path)
    
    with pymupdf.open(file_path) as doc:
        pages = list(doc)
        
        # Hàm xử lý cho từng trang riêng biệt
        def process_single_page(page):
            page_number = page.number + 1
            # 1. Trích xuất text trực tiếp (Cực nhanh)
            raw_text = page.get_text("text")
            text = clean_ocr_text(raw_text)

            # 2. Kiểm tra điều kiện chạy OCR
            # Chỉ chạy khi người dùng cho phép (use_ocr=True) VÀ text trực tiếp không đủ tốt
            if use_ocr and not is_text_good_enough(text):
                ocr_text = extract_text_with_ocr(page)
                if len(ocr_text) > 0:
                    text = ocr_text
            
            page_chunks = []
            if text:
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    page_chunks.append({
                        "page_content": chunk,
                        "metadata": {
                            "source": source_name, 
                            "page": page_number, 
                            "type": "text"
                        }
                    })
            return page_chunks

        # Xử lý song song các trang để tận dụng nhân CPU
        all_chunks = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single_page, pages))
            
        for page_result in results:
            all_chunks.extend(page_result)

    # Đánh lại chunk_id sau khi gom đủ từ tất cả các trang
    for i, chunk in enumerate(all_chunks):
        chunk["metadata"]["chunk_id"] = f"c{i+1:03}"

    return all_chunks, []


def process_docx(file_path: str, chunk_size: int, chunk_overlap: int):
    """Đọc và băm nhỏ file Word DOCX (Không cần OCR)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc = Document(file_path)

    full_text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    ).strip()
    all_chunks = []

    if full_text:
        text_chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(text_chunks, start=1):
            all_chunks.append(
                {
                    "page_content": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": 1,
                        "chunk_id": f"c{idx:03}",
                        "type": "text",
                    },
                }
            )

    return all_chunks, []