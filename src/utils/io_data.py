import json

def load_json_data(path_file):
    """
    Hàm load data
    Return: Mảng với mỗi phần tử có dạng {"words": [], "tags": []}
    """
    data = []
    with open(path_file, mode="r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def save_json_data(data, save_path):
    """
    Hàm save data vào file json
    Dữ liệu đầu vào dạng mảng [{"words": [], "tags": []}, ]
    """
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in data:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")