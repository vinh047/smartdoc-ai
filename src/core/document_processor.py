import os
import re
import fitz
import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageFilter, ImageOps
from docx import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument

from src.config import TEXT_SPLITTER_CONFIG, EMBEDDING_CONFIG
from src.core.metadata_handler import MetadataManager



def process_document(temp_path, chunk_size=None, chunk_overlap=None):
    """
    Hàm chính dùng cho luồng hiện tại của app.
    Trả về FAISS vector store để tương thích với rag_engine.py.

    Flow:
    - Kiểm tra file
    - Xử lý PDF/DOCX -> chunks, images
    - Build vector store từ chunks
    """
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Không tìm thấy file: {temp_path}")

    file_extension = os.path.splitext(temp_path)[1].lower()

    if chunk_size is None:
        chunk_size = TEXT_SPLITTER_CONFIG["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = TEXT_SPLITTER_CONFIG["chunk_overlap"]

    if file_extension == ".pdf":
        chunks, images = process_pdf(temp_path, chunk_size, chunk_overlap)
    elif file_extension == ".docx":
        chunks, images = process_docx(temp_path, chunk_size, chunk_overlap)
    else:
        raise ValueError("Chỉ hỗ trợ file PDF và DOCX")

    # Có thể lưu images ra session/db sau nếu cần
    # Hiện tại để không phá luồng cũ, chỉ build vector store từ text chunks
    vector_store = build_vector_store(chunks)
    return vector_store


def process_document_data(temp_path, chunk_size=None, chunk_overlap=None):
    """
    Hàm phụ nếu bạn muốn lấy trực tiếp dữ liệu chunks + images
    cho TV2/TV4 hoặc debug.
    """
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Không tìm thấy file: {temp_path}")

    file_extension = os.path.splitext(temp_path)[1].lower()

    if chunk_size is None:
        chunk_size = TEXT_SPLITTER_CONFIG["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = TEXT_SPLITTER_CONFIG["chunk_overlap"]

    if file_extension == ".pdf":
        return process_pdf(temp_path, chunk_size, chunk_overlap)
    elif file_extension == ".docx":
        return process_docx(temp_path, chunk_size, chunk_overlap)
    else:
        raise ValueError("Chỉ hỗ trợ file PDF và DOCX")


def build_vector_store(chunks):
    """
    Chuyển danh sách chunk dict -> LangChain Document -> FAISS
    """
    if not chunks:
        raise ValueError("Không trích xuất được nội dung văn bản từ tài liệu.")

    documents = [
        LangchainDocument(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        for item in chunks
    ]

    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={"device": EMBEDDING_CONFIG["device"]},
        encode_kwargs={"normalize_embeddings": EMBEDDING_CONFIG["normalize_embeddings"]}
    )

    vector_store = FAISS.from_documents(documents, embedder)
    return vector_store


def clean_ocr_text(text: str) -> str:
    """
    Làm sạch text sau OCR / text extract trực tiếp.
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Bỏ các dòng quá nhiều ký tự rác
        non_alnum = len(re.sub(r"[A-Za-zÀ-ỹ0-9]", "", line))
        if len(line) > 0 and non_alnum > len(line) * 0.7:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def is_text_good_enough(text: str) -> bool:
    """
    Đánh giá text extract trực tiếp đã đủ tốt chưa.
    Nếu chưa tốt thì mới OCR để tiết kiệm thời gian.
    """
    if not text:
        return False

    text = text.strip()
    if len(text) < 80:
        return False

    alnum_count = sum(ch.isalnum() for ch in text)
    ratio = alnum_count / max(len(text), 1)

    return ratio > 0.45


def render_page_to_pil(page, dpi=220):
    """
    Render 1 trang PDF thành ảnh PIL.
    DPI 220 là mức cân bằng giữa tốc độ và độ rõ.
    """
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Tiền xử lý ảnh để OCR tốt hơn:
    - grayscale
    - autocontrast
    - upscale nhẹ nếu nhỏ
    - sharpen
    - median blur
    - adaptive threshold
    """
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img)

    w, h = img.size
    if w < 1400:
        scale = 1.5
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS
        )

    img = img.filter(ImageFilter.SHARPEN)

    img_np = np.array(img)
    img_np = cv2.medianBlur(img_np, 3)

    img_np = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return Image.fromarray(img_np)


def extract_text_with_ocr(page, lang="vie+eng"):
    """
    OCR một trang PDF bằng Tesseract.
    Thử config 1 trước, nếu text yếu thì fallback config 2.
    """
    pil_img = render_page_to_pil(page, dpi=220)
    processed_img = preprocess_image_for_ocr(pil_img)

    config_1 = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(
        processed_img,
        lang=lang,
        config=config_1
    )
    text = clean_ocr_text(text)

    if len(text) < 40:
        config_2 = r"--oem 3 --psm 4"
        text2 = pytesseract.image_to_string(
            processed_img,
            lang=lang,
            config=config_2
        )
        text2 = clean_ocr_text(text2)

        if len(text2) > len(text):
            text = text2

    return text


def process_pdf(file_path, chunk_size, chunk_overlap):
    """
    Xử lý PDF:
    - đọc text trực tiếp
    - nếu text yếu thì fallback OCR
    - chunk text
    - trích ảnh từ PDF
    - trả về (all_chunks, all_images)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    all_images = []

    source_name = os.path.basename(file_path)
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)

    chunk_counter = 1

    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1

            # 1. Đọc text trực tiếp từ PDF
            text = clean_ocr_text(page.get_text("text"))

            # 2. Nếu text yếu, dùng OCR fallback
            if not is_text_good_enough(text):
                try:
                    ocr_text = extract_text_with_ocr(page)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception as e:
                    print(f"[OCR WARNING] Trang {page_number}: {e}")

            # 3. Chunk text
            if text:
                text_chunks = splitter.split_text(text)
                for chunk in text_chunks:
                    all_chunks.append({
                        "page_content": chunk,
                        "metadata": {
                            "source": source_name,
                            "page": page_number,
                            "chunk_id": f"c{chunk_counter:03}",
                            "type": "text"
                        }
                    })
                    chunk_counter += 1

            # 4. Trích ảnh từ PDF
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images, start=1):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_filename = (
                        f"{os.path.splitext(source_name)[0]}_p{page_number}_img{img_idx}.{image_ext}"
                    )
                    image_path = os.path.join(image_dir, image_filename)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    all_images.append({
                        "image_path": image_path,
                        "metadata": {
                            "source": source_name,
                            "page": page_number,
                            "type": "image"
                        }
                    })
                except Exception as e:
                    print(f"[IMAGE WARNING] Trang {page_number}, ảnh {img_idx}: {e}")

    return all_chunks, all_images


def process_docx(file_path, chunk_size, chunk_overlap):
    """
    Xử lý DOCX:
    - đọc paragraph
    - nối text
    - chunk
    - trả về (all_chunks, [])
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    doc = Document(file_path)

    full_text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    ).strip()

    all_chunks = []
    source_name = os.path.basename(file_path)

    if full_text:
        text_chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(text_chunks, start=1):
            all_chunks.append({
                "page_content": chunk,
                "metadata": {
                    "source": source_name,
                    "page": 1,
                    "chunk_id": f"c{idx:03}",
                    "type": "text"
                }
            })

    return all_chunks, []