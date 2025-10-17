import unicodedata
import re

from src.utils.io_data import load_json_data, save_json_data
def normalize_text(text: str):
    """
    Chuẩn hóa chuỗi đầu vào, xóa khoảng trắng thừa
    """
    text = text.strip()
    text = unicodedata.normalize("NFC", text)

    return text


def preprocess_data(source_path, save_path):
    """
    Chuẩn hóa dữ liệu NER:
      - Normalize Unicode + xử lý dấu câu/khoảng trắng
    """
    data = load_json_data(source_path)

    processed = []
    for i, sample in enumerate(data):
        words = [normalize_text(text) for text in sample['words']]
        tags = sample['tags']

        if len(words) != len(tags):
            print(f"Warning: số token và tag không khớp — bỏ qua sample: {sample}")

        processed.append({"words": words, "tags": tags})

    save_json_data(processed, save_path)
