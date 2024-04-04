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

class ttest(IFeatureExtractor):
    def fit_transform(self, X, y, **kwargs):
        sfeatures = find_significant_features(X, y, kwargs.get(alpha, 1e-4))
        return X[sfeatures]

    def find_significant_features(feature_table, class_table, alpha):
        classes = class_table['Study.Group'].unique()
        significant_features = []

        for feature in feature_table.columns:
            class_1_values = feature_table[class_table['Study.Group'] == classes[0]][feature]
            class_2_values = feature_table[class_table['Study.Group'] == classes[1]][feature]

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(class_1_values, class_2_values)

            if p_value < alpha:  # You can adjust the significance level as needed
                significant_features.append(feature)

        return significant_features

class oneway(IFeatureExtractor):

    def fit_transform(self, X, y, **kwargs):
        sfeatures = find_significant_features(X, y, kwargs.get(alpha, 1e-4))
        return X[sfeatures]

    def find_significant_features(feature_table, class_table, alpha):
        # Assuming class_table has a column 'class' indicating the classes
        classes = class_table['Study.Group'].unique()
        significant_features = []

        for feature in feature_table.columns:
            # List to store values for each class
            class_values = []
            for class_label in classes:
                class_values.append(feature_table[class_table['Study.Group'] == class_label][feature])

            # Perform ANOVA test
            f_stat, p_value = stats.f_oneway(*class_values)

            # Check significance (e.g., p-value < 0.05)
            if p_value < alpha:  # You can adjust the significance level as needed
                significant_features.append(feature)
                print(p_value)

        return significant_features

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
