from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
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


class DataCleaning:
    @staticmethod
    def remove_na(df, threshold):
        return df.dropna(thresh=threshold, axis=1)
    
    def remove_duplicates(df):
        return df.drop_duplicates()
    
    def remove_outliers(df):
        return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    
    def remove_low_variance(df, threshold):
        return df.loc[:, df.var() > threshold]
    
    def remove_correlated_features(df, threshold):
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(to_drop, axis=1)
    
    def remove_low_importance_features(df, model, threshold):
        model.fit(df)
        importance = model.feature_importances_
        return df.loc[:, importance > threshold]
    
    def clean_data(genera, metadata, mtb, na_threshold):
        mtb = mtb.dropna(thresh=mtb.shape[0] * na_threshold, axis=1)
        mtb = mtb.dropna(thresh=mtb.shape[1] * na_threshold, axis=0)
        metadata = metadata[metadata['Sample'].isin(mtb.Sample)]
        genera = genera[genera['Sample'].isin(mtb.Sample)]
        return genera, metadata, mtb