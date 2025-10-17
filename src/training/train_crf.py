from src.models.crf_model import CRFModel
from src.data.feature_extractor import sent2features, sent2labels
from src.utils.io_data import load_json_data


def train(train_path, test_path, dev_path):
    train_data = load_json_data(train_path)
    dev_data = load_json_data(dev_path)
    test_data = load_json_data(test_path)

    X_train = [sent2features(sample['words']) for sample in train_data]
    y_train = [sent2labels(sample['tags']) for sample in train_data]

    X_dev = [sent2features(sample['words']) for sample in dev_data]
    y_dev = [sent2labels(sample['tags']) for sample in dev_data]

    X_test = [sent2features(sample['words']) for sample in test_data]
    y_test = [sent2labels(sample['tags']) for sample in test_data]


    model = CRFModel(c1=0.1, c2=0.1, algorithm='lbfgs', max_iterations=200)
    model.train(X_train, y_train)
    model.evaluate(X_train, y_train)
    model.evaluate(X_dev, y_dev)
    model.evaluate(X_test, y_test)


train("data/raw/PhoNER_COVID19/word/train_word.json", 
      "data/raw/PhoNER_COVID19/word/test_word.json", 
      "data/raw/PhoNER_COVID19/word/dev_word.json")