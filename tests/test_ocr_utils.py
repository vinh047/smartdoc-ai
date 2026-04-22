import pytest
from src.core.ocr_utils import clean_ocr_text, is_text_good_enough

def test_clean_ocr_text():
    """Kiểm tra xem hàm có dọn sạch các khoảng trắng và ký tự xuống dòng thừa không"""
    dirty_text = "Vinamilk   công bố    \n\n\n lợi nhuận."
    expected = "Vinamilk công bố\n\nlợi nhuận."
    assert clean_ocr_text(dirty_text) == expected

def test_is_text_good_enough():
    """Kiểm tra logic nhận diện văn bản rác"""
    good_text = "Đây là một đoạn văn bản rất dài và rõ ràng, có đầy đủ các ký tự hợp lệ để hệ thống đọc hiểu." * 2
    bad_text = "@@@ #### $%^ &*()_+ !!! ~~~" * 10
    
    assert is_text_good_enough(good_text) == True
    assert is_text_good_enough(bad_text) == False