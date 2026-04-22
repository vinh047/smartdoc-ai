import os
import re
import tempfile
import easyocr

# Khởi tạo mô hình EasyOCR 1 lần duy nhất khi file được import
reader = easyocr.Reader(["vi", "en"], gpu=True)


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


def extract_text_with_ocr(page) -> str:
    """
    Trích xuất văn bản từ trang PDF bằng AI Thị giác (EasyOCR).
    Đã xử lý chống lỗi Permission Denied trên Windows.
    """
    pix = page.get_pixmap(dpi=220)

    # Tạo file tạm và đóng ngay để chống Lock file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()

    try:
        pix.save(tmp_path)
        print("🔥 Đang quét ảnh bằng EasyOCR...")
        result = reader.readtext(tmp_path, detail=0, paragraph=True)
        return "\n".join(result)
    except Exception as e:
        print(f"❌ Lỗi EasyOCR: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
