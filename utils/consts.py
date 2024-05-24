from analysis.preprocessing import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

DEFAULT_DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
OUTPUT_LOG_FILE = 'log.txt'
DEFAULT_DEBUG_MODE = False

NA_THRESHOLD = 0.90
RANDOM_STATE = 42
KFOLD_SPLITS = 5
DEFAULT_SCORING = 'f1_weighted'

ANALYSIS_PARAMS = {
    'mars_ibs_lr': {
        'dir': 'MARS_IBS_2020',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': LogisticRegression,
        'classifier_param_grid': {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 300],
            'tol': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        },
    },
    'mars_ibs_rf': {
        'dir': 'MARS_IBS_2020',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': RandomForestClassifier,
        'classifier_param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        },
    },
    'mars_ibs_mlp': {
        'dir': 'MARS_IBS_2020',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': MLPClassifier,
        'classifier_param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
    },
    'ihmp_ibdmdb_lr': {
        'dir': 'iHMP_IBDMDB_2019',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': LogisticRegression,
        'classifier_param_grid': {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 300],
            'tol': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        },
    },
    'ihmp_ibdmdb_rf': {
        'dir': 'iHMP_IBDMDB_2019',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': RandomForestClassifier,
        'classifier_param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        },
    },
    'erawijantari_lr': {
        'dir': 'Erawijantari',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': LogisticRegression,
        'pos_label': 'Gastrectomy',
        'classifier_param_grid': {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 300],
            'tol': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        },
    },
    'erawijantari_rf': {
        'dir': 'Erawijantari',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': RandomForestClassifier,
        'classifier_param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        },
    },
    'franzosa_lr': {
        'dir': 'franzosa',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': LogisticRegression,
        'classifier_param_grid': {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 300],
            'tol': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced'],
            'random_state': [42]
        },
    },
    'franzosa_rf': {
        'dir': 'franzosa',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': RandomForestClassifier,
        'classifier_param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        },
    },
    'franzosa_mlp': {
        'dir': 'franzosa',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': MLPClassifier,
        'classifier_param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
    },
}
