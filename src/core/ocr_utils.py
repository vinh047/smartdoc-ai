from datetime import datetime
import os
import re
import tempfile
import easyocr

# Khởi tạo mô hình EasyOCR 1 lần duy nhất khi file được import
reader = easyocr.Reader(["vi", "en"], gpu=False)


def clean_ocr_text(text: str) -> str:
    """
    Làm sạch văn bản thô: xóa khoảng trắng thừa, xóa dòng rác.
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Bỏ các dòng quá nhiều ký tự rác (không phải chữ/số)
        non_alnum = len(re.sub(r"[A-Za-zÀ-ỹ0-9]", "", line))
        if len(line) > 0 and non_alnum > len(line) * 0.7:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_text_good_enough(text: str) -> bool:
    """
    Đánh giá xem văn bản đọc trực tiếp từ PDF có đủ tốt không.
    Nếu tỷ lệ chữ/số < 45% -> Rác -> Cần dùng OCR.
    """
    if not text or len(text.strip()) < 80:
        return False

    eval_text = re.sub(r"[\.\-\_]", "", text.strip())
    if len(eval_text) == 0:
        return False

    alnum_count = sum(ch.isalnum() for ch in text)
    return (alnum_count / max(len(text), 1)) > 0.45


def extract_text_with_ocr(page, page_index: int = 0) -> str:
    """
    Trích xuất văn bản từ trang PDF bằng AI Thị giác và LƯU LẠI ẢNH để kiểm tra.
    """
    # 1. Tạo thư mục lưu trữ nếu chưa có
    save_dir = "ocr_logs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Tạo tên file theo thời gian và số trang để tránh trùng lặp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"page_{page_index + 1}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    # 3. Chuyển trang PDF thành ảnh (DPI 220 là mức cân bằng giữa nét và nhẹ)
    pix = page.get_pixmap(dpi=220)

    try:
        # 4. Lưu ảnh vĩnh viễn vào thư mục ocr_logs
        pix.save(save_path)
        print(f"📸 Đã lưu ảnh trang {page_index + 1} tại: {save_path}")
        
        print("🔥 Đang quét ảnh bằng EasyOCR...")
        # Đọc text từ file ảnh vừa lưu
        result = reader.readtext(save_path, detail=0, paragraph=True)
        return "\n".join(result)
        
    except Exception as e:
        print(f"❌ Lỗi EasyOCR: {e}")
        return ""