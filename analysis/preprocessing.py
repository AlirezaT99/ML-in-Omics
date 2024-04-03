from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import chi2

class IFeatureExtractor(ABC):
	def __init__(self):
		self._model = None

	@abstractmethod
	def fit_transform(self, X, y, **kwargs):
		raise NotImplementedError

class PC(IFeatureExtractor):
    def fit_transform(self, X, y, **kwargs):
        min_var = kwargs.get('var', 0.9)
        pca = PCA(n_components=min_var)
        return pca.fit_transform(X)
	
class LDA(IFeatureExtractor):
    def fit_transform(self, X, y, **kwargs):
        X = X.toarray()
        lda = LinearDiscriminantAnalysis()
        return lda.fit_transform(X, y)

class Chi2(IFeatureExtractor):
    def fit_transform(self, X, y, **kwargs):
        chi2_stats, _ = chi2(X, y)
        k = kwargs.get('k', 100)
        indices = np.argsort(chi2_stats)[::-1][:k]
        return X[:, indices]

