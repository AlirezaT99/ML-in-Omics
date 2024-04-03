from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class IMetaModel(ABC):
	def __init__(self):
		self._model = None

	@abstractmethod
	def train_model(self, X_train, y_train, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def predict(self, X_test):
		raise NotImplementedError


class RF(IMetaModel):
	def train_model(self, X_train, y_train, model_kwargs):
		fixed_hyperparams = {
			'random_state': 10,
		}

		self._model = RandomForestClassifier(**model_kwargs, **fixed_hyperparams)
		self._model.fit(X_train, y_train)
	
	def predict(self, X_test):
		assert self._model, 'Model is not loaded/trained yet'
		return self._model.predict(X_test)


class LogisticRegression(IMetaModel):
	model = LogisticRegression()
	pass  # TODO
