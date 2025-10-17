from string import punctuation


import string

def word2features(words, position):
    """
    Chuyển một token thành vector đặc trưng cho CRF.
    Args:
        words: danh sách token trong câu
        position: vị trí của token hiện tại
    """
    word = words[position]

    # Đặc trưng cơ bản
    features = {
        "bias": 1.0,
        "word": word,
        "word.lower": word.lower(),
        "word.is_first": position == 0,
        "word.is_last": position == len(words) - 1,
        "word.is_capitalized": word[0].isupper(),
        "word.is_all_caps": word.isupper(),
        "word.is_all_lower": word.islower(),
        "word.prefix2": word[:2],
        "word.suffix2": word[-2:],
        "word.contains_digit": any(ch.isdigit() for ch in word),
        "word.contains_dash": "-" in word,
        "word.contains_underscore": "_" in word,
        "word.is_punct": word in string.punctuation,
        "word.length": len(word),
    }

    # Đặc trưng ngữ cảnh
    if position > 0:
        prev = words[position - 1]
        features.update({
            "-1:word": prev,
            "-1:lower": prev.lower(),
            "-1:istitle": prev.istitle(),
            "-1:isupper": prev.isupper(),
            "-1:ispunct": prev in string.punctuation,
        })
    else:
        features["BOS"] = True  # Begin of sentence

    if position > 1:
        prev2 = words[position - 2]
        features.update({
            "-2:lower": prev2.lower(),
            "-2:istitle": prev2.istitle(),
            "-2:isupper": prev2.isupper(),
        })

    if position < len(words) - 1:
        nxt = words[position + 1]
        features.update({
            "+1:word": nxt,
            "+1:lower": nxt.lower(),
            "+1:istitle": nxt.istitle(),
            "+1:isupper": nxt.isupper(),
            "+1:ispunct": nxt in string.punctuation,
        })
    else:
        features["EOS"] = True  # End of sentence

    if position < len(words) - 2:
        nxt2 = words[position + 2]
        features.update({
            "+2:lower": nxt2.lower(),
            "+2:istitle": nxt2.istitle(),
            "+2:isupper": nxt2.isupper(),
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(tags):
    return tags