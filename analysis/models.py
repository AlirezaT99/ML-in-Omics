from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class IModel(ABC):
    def __init__(self):
        self._model = None

    @abstractmethod
    def train_model(self, X_train, y_train, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError


class RF(IModel):
    def train_model(self, X_train, y_train, model_kwargs):
        fixed_hyperparams = {
            'random_state': 10,
        }

        self._model = RandomForestClassifier(**model_kwargs, **fixed_hyperparams)
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        assert self._model, 'Model is not loaded/trained yet'
        return self._model.predict(X_test)


class LR(LogisticRegression):
    def __init__(self, **kwargs):
        super(LR, self).__init__(**kwargs)
