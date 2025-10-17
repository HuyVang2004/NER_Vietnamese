import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics

class CRFModel:
    def __init__(self, c1=0.1, c2=0.1, algorithm='lbfgs', max_iterations=100):
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1, c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted')
        print(f"F1-score: {f1:.4f}")
        print(metrics.flat_classification_report(y_test, y_pred))
        return f1

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
